"""AirInk: raise only your index finger to write."""

import os
import sys
import time
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk

WIDTH, HEIGHT = 960, 540
ROOT = Path(__file__).resolve().parent
COLORS = ("#2563eb", "#172033", "#e34b65", "#169b83", "#e7a52b", "#8b5cf6")
BG, PANEL, TEXT, MUTED = "#f3f5f9", "white", "#172033", "#67758a"


def storage_folder():
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path.home() / ".local" / "share"
    return base / "AirInk" / "drawings"


def writing_pose(hand):
    """Write only with the index finger; keep every other finger folded."""
    raised = [hand[tip].y < hand[tip - 2].y - 0.025 for tip in (8, 12, 16, 20)]

    def distance(a, b):
        return np.hypot(hand[a].x - hand[b].x, hand[a].y - hand[b].y)

    thumb_open = distance(4, 17) > 1.15 * distance(3, 17)
    return raised == [True, False, False, False] and not thumb_open


class OneEuro:
    """1€ filter (Casiez et al., 2012): strong smoothing when slow, little lag when fast."""

    def __init__(self, min_cutoff=1.0, beta=0.01, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self.x = self.speed = self.t = None

    def __call__(self, point, t):
        x = np.array(point, dtype=float)
        if self.x is None:
            self.x, self.speed, self.t = x, np.zeros(2), t
            return tuple(x)
        dt, self.t = max(t - self.t, 1e-3), t

        def alpha(cutoff):
            return 1 / (1 + 1 / (2 * np.pi * cutoff * dt))

        self.speed += alpha(self.d_cutoff) * ((x - self.x) / dt - self.speed)
        self.x += alpha(self.min_cutoff + self.beta * np.linalg.norm(self.speed)) * (x - self.x)
        return tuple(self.x)


class AirInk:
    def __init__(self, root):
        self.root, self.camera, self.tracker = root, None, None
        self.stroke, self.pose_frames, self.missed = [], 0, 0
        self.timestamp, self.failed_reads, self.after_id = 0, 0, None
        self.smoother, self.color = OneEuro(), COLORS[0]
        self.ink = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)   # colour of each pixel
        self.mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)     # 0 = empty, 255 = ink
        self.crop = (0, 0, 1, 1)
        self.tool, self.view = tk.StringVar(value="Pen"), tk.StringVar(value="Camera")
        self.size, self.opacity = tk.IntVar(value=6), tk.IntVar(value=60)
        self.status = tk.StringVar(value="Camera off")
        root.title("AirInk")
        root.configure(bg=BG)
        root.resizable(False, False)
        self.build_ui()
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<Escape>", lambda event: self.close())
        self.tick()

    # ---------- interface ----------
    def build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", font=("Helvetica", 11), padding=(10, 9),
                        background="#edf1f7", foreground=TEXT, borderwidth=0)
        style.map("TButton", background=[("active", "#dfe7f4")])
        style.configure("Accent.TButton", background="#2563eb", foreground="white")
        style.map("Accent.TButton", background=[("active", "#1e4aa3")])
        style.configure("TRadiobutton", background=PANEL, font=("Helvetica", 11))
        style.configure("TScale", background="#2563eb", troughcolor="#edf1f7", borderwidth=0)

        header = tk.Frame(self.root, bg=BG)
        header.pack(fill="x", padx=24, pady=(20, 16))
        tk.Label(header, text="AirInk", font=("Helvetica", 26, "bold"), bg=BG, fg=TEXT).pack(side="left")
        tk.Label(header, textvariable=self.status, font=("Helvetica", 11),
                 bg="#e5eaf4", fg="#35527c", padx=14, pady=7).pack(side="right")

        body = tk.Frame(self.root, bg=BG)
        body.pack(padx=24)
        self.screen = tk.Label(body, bd=0, bg=PANEL)
        self.screen.pack(side="left")
        self.screen.bind("<Button-1>", self.mouse_draw)
        self.screen.bind("<B1-Motion>", self.mouse_draw)
        self.screen.bind("<ButtonRelease-1>", lambda event: self.reset_stroke())
        panel = tk.Frame(body, bg=PANEL, width=196, height=HEIGHT)
        panel.pack(side="left", fill="y", padx=(16, 0))
        panel.pack_propagate(False)

        def title(text):
            tk.Label(panel, text=text, bg=PANEL, fg=MUTED, font=("Helvetica", 10, "bold"),
                     anchor="w").pack(fill="x", padx=18, pady=(14, 6))

        def choices(variable, names):
            row = tk.Frame(panel, bg=PANEL)
            row.pack(fill="x", padx=16)
            for name in names:
                ttk.Radiobutton(row, text=name, value=name, variable=variable,
                                command=self.reset_stroke).pack(side="left")

        self.camera_button = ttk.Button(panel, text="Start camera", style="Accent.TButton",
                                        command=self.toggle_camera)
        self.camera_button.pack(fill="x", padx=16, pady=(18, 0))
        title("TOOL")
        choices(self.tool, ("Pen", "Eraser"))
        title("COLOUR")
        row = tk.Frame(panel, bg=PANEL)
        row.pack(padx=16)
        self.swatches = {}
        for index, color in enumerate(COLORS):
            swatch = tk.Label(row, bg=color, width=2, height=1, bd=3,
                              relief="solid" if color == self.color else "flat")
            swatch.grid(row=0, column=index, padx=2, pady=4)
            swatch.bind("<Button-1>", lambda event, c=color: self.choose_color(c))
            self.swatches[color] = swatch
        title("BRUSH SIZE")
        ttk.Scale(panel, from_=2, to=24, variable=self.size).pack(fill="x", padx=18)
        title("VIEW")
        choices(self.view, ("Camera", "Board"))
        title("CAMERA OPACITY")
        ttk.Scale(panel, from_=0, to=100, variable=self.opacity).pack(fill="x", padx=18)

        row = tk.Frame(panel, bg=PANEL)
        row.pack(side="bottom", fill="x", padx=16, pady=(0, 18))
        ttk.Button(row, text="Open", width=5, command=self.open_drawing).pack(side="left", expand=True, fill="x")
        ttk.Button(row, text="Save", width=5, style="Accent.TButton", command=self.save).pack(
            side="left", expand=True, fill="x", padx=(6, 0))
        ttk.Button(panel, text="Clear", command=self.clear).pack(side="bottom", fill="x", padx=16, pady=(0, 8))
        tk.Label(self.root, text="Only index up: draw     •     More fingers: pause     •     Camera off: use mouse",
                 font=("Helvetica", 11), bg=BG, fg=MUTED).pack(pady=16)

    def choose_color(self, color):
        self.reset_stroke()
        self.color = color
        self.tool.set("Pen")
        for value, swatch in self.swatches.items():
            swatch.configure(relief="solid" if value == color else "flat")

    # ---------- camera ----------
    def toggle_camera(self):
        if self.camera is not None:
            return self.stop_camera()
        camera = tracker = None
        try:
            vision = mp.tasks.vision
            tracker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(ROOT / "assets" / "hand_landmarker.task")),
                running_mode=vision.RunningMode.VIDEO, num_hands=1, min_hand_detection_confidence=0.6,
                min_hand_presence_confidence=0.6, min_tracking_confidence=0.6))
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                raise RuntimeError("Camera unavailable. Check camera permission or close other camera apps.")
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        except Exception as error:
            if camera is not None:
                camera.release()
            if tracker is not None:
                tracker.close()
            messagebox.showerror("Camera could not start", str(error))
            return
        self.camera, self.tracker = camera, tracker
        self.pose_frames = self.failed_reads = 0
        self.reset_stroke()
        self.camera_button.configure(text="Stop camera")
        self.status.set("Show your hand")

    def stop_camera(self):
        if self.camera is not None:
            self.camera.release()
        if self.tracker is not None:
            self.tracker.close()
        self.camera = self.tracker = None
        self.pose_frames = 0
        self.smoother.reset()
        self.reset_stroke()
        self.camera_button.configure(text="Start camera")
        self.status.set("Camera off")

    def camera_view(self, frame):
        """Centre-crop the camera to 16:9 so it is never stretched."""
        height, width = frame.shape[:2]
        crop_w, crop_h = round(min(width, height * WIDTH / HEIGHT)), round(min(height, width * HEIGHT / WIDTH))
        left, top = (width - crop_w) // 2, (height - crop_h) // 2
        self.crop = (left / width, top / height, crop_w / width, crop_h / height)
        return cv2.resize(frame[top:top + crop_h, left:left + crop_w], (WIDTH, HEIGHT))

    def view_point(self, tip):
        """Map a fingertip (0–1 camera coordinates) onto the screen, or None if outside it."""
        left, top, width, height = self.crop
        x, y = (tip.x - left) / width, (tip.y - top) / height
        return (x * (WIDTH - 1), y * (HEIGHT - 1)) if 0 <= x <= 1 and 0 <= y <= 1 else None

    # ---------- drawing ----------
    def reset_stroke(self):
        """End the stroke: finish the last half-segment, then start fresh."""
        if len(self.stroke) >= 2:
            self.paint([(self.stroke[-2] + self.stroke[-1]) / 2, self.stroke[-1]])
        self.stroke = []

    def draw(self, point, camera=True):
        """Add a point to the stroke and draw a smooth curve through the last three points."""
        point = np.clip(np.array(point, dtype=float), 0, (WIDTH - 1, HEIGHT - 1))
        if camera and self.stroke and np.linalg.norm(point - self.stroke[-1]) > 120:
            self.reset_stroke()  # tracking jump: don't draw a bridge
        self.stroke = (self.stroke + [point])[-3:]
        if len(self.stroke) == 1:
            self.paint([point, point])  # a single dot
        elif len(self.stroke) == 2:
            self.paint([self.stroke[0], (self.stroke[0] + point) / 2])
        else:  # quadratic Bézier from midpoint to midpoint, bending through the middle point
            a, b, c = self.stroke
            t = np.linspace(0, 1, 12)[:, None]
            self.paint((1 - t) ** 2 * (a + b) / 2 + 2 * (1 - t) * t * b + t ** 2 * (b + c) / 2)

    def paint(self, curve):
        line = [np.round(np.array(curve) * 16).astype(np.int32)]  # 4-bit sub-pixel precision
        erase = self.tool.get() == "Eraser"
        width = max(24, self.size.get() * 4) if erase else max(2, self.size.get())
        if not erase:
            color = tuple(int(self.color[i:i + 2], 16) for i in (1, 3, 5))
            cv2.polylines(self.ink, line, False, color, width + 2, cv2.LINE_8, shift=4)
        cv2.polylines(self.mask, line, False, 0 if erase else 255, width, cv2.LINE_AA, shift=4)

    def mouse_draw(self, event):
        if self.camera is None:
            self.draw((event.x, event.y), camera=False)

    def compose(self, background=None):
        """Blend ink over the camera (faded by opacity) or over white."""
        white = np.full_like(self.ink, 255)
        if background is None:
            background = white
        else:
            opacity = self.opacity.get() / 100
            background = cv2.addWeighted(background, opacity, white, 1 - opacity, 0)
        alpha = self.mask[:, :, None].astype(np.uint16)
        return ((self.ink * alpha + background * (255 - alpha)) // 255).astype(np.uint8)

    # ---------- main loop ----------
    def tick(self):
        background = point = None
        try:
            if self.camera is not None:
                background, point = self.camera_step()
        except Exception as error:
            self.stop_camera()
            messagebox.showerror("Camera stopped", str(error))
        try:
            picture = self.compose(background)
            if self.camera is not None and point is not None:
                cv2.circle(picture, tuple(int(v) for v in point), 10, (22, 155, 131), 2, cv2.LINE_AA)
            self.photo = ImageTk.PhotoImage(Image.fromarray(picture))
            self.screen.configure(image=self.photo)
        finally:
            self.after_id = self.root.after(15, self.tick)

    def camera_step(self):
        """Read one frame, track the hand, draw if the pose holds; return (background, point)."""
        ok, frame = self.camera.read()
        if not ok:  # macOS often returns empty frames while the camera warms up
            self.failed_reads += 1
            if self.failed_reads > 100:
                raise RuntimeError("Camera disconnected. You can continue with the mouse.")
            self.status.set("Starting camera…")
            return None, None
        self.failed_reads = 0
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        self.timestamp = max(self.timestamp + 1, time.monotonic_ns() // 1_000_000)
        result = self.tracker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame),
                                               self.timestamp)
        preview = self.camera_view(frame)
        hand = result.hand_landmarks[0] if result.hand_landmarks else None
        point = self.view_point(hand[8]) if hand else None
        if point is None:
            self.smoother.reset()
        else:
            point = self.smoother(point, self.timestamp / 1000)
        writing = point is not None and writing_pose(hand)
        # Start after 3 good frames; stop after 3 missed ones, so one bad frame won't split a line.
        self.missed = 0 if writing else self.missed + 1
        if writing:
            self.pose_frames += 1
        elif self.missed >= 3:
            self.pose_frames = 0
        drawing = writing and self.pose_frames >= 3
        if drawing:
            self.draw(point)
        elif self.pose_frames < 3:
            self.reset_stroke()
        self.status.set(("Drawing" if self.tool.get() == "Pen" else "Erasing") if drawing else
                        "Move / pause" if hand else "Show your hand")
        return (preview if self.view.get() == "Camera" else None), point

    # ---------- files ----------
    def clear(self):
        self.ink.fill(0)
        self.mask.fill(0)
        self.pose_frames = 0
        self.reset_stroke()

    def save(self):
        try:
            folder = storage_folder()
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"AirInk-{datetime.now():%Y%m%d-%H%M%S-%f}.png"
            Image.fromarray(self.compose()).save(path, format="PNG")
            self.root.title(f"AirInk — Saved {path.name}")
        except OSError as error:
            messagebox.showerror("Could not save", str(error))

    def open_drawing(self):
        try:
            folder = storage_folder()
            folder.mkdir(parents=True, exist_ok=True)
            path = filedialog.askopenfilename(title="Open saved drawing", initialdir=str(folder),
                                              filetypes=[("PNG drawings", "*.png")])
            if not path:
                return
            with Image.open(path) as image:
                if image.format != "PNG" or image.size != (WIDTH, HEIGHT):
                    raise ValueError("Choose a 960 × 540 PNG saved by AirInk.")
                rgba = image.convert("RGBA")
                self.ink = np.array(Image.alpha_composite(Image.new("RGBA", image.size, "white"), rgba).convert("RGB"))
            self.mask = np.any(self.ink != 255, axis=2).astype(np.uint8) * 255
            self.pose_frames = 0
            self.reset_stroke()
            self.root.title(f"AirInk — {Path(path).name}")
        except (OSError, ValueError) as error:
            messagebox.showerror("Could not open", str(error))

    def close(self):
        if self.after_id is None:
            return
        self.root.after_cancel(self.after_id)
        self.after_id = None
        self.stop_camera()
        self.root.destroy()


if __name__ == "__main__":
    AirInk(tk.Tk()).root.mainloop()
