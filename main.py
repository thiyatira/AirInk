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


class AirInk:
    def __init__(self, root):
        self.root, self.camera, self.tracker = root, None, None
        self.previous, self.pose_frames, self.timestamp = None, 0, 0
        self.color = COLORS[0]
        self.ink = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        self.mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        self.tool = tk.StringVar(value="Pen")
        self.view = tk.StringVar(value="Camera")
        self.size = tk.IntVar(value=6)
        self.opacity = tk.IntVar(value=60)
        self.zoom = tk.DoubleVar(value=1.5)
        self.crop = (0, 0, 1, 1)
        self.status = tk.StringVar(value="Camera off")
        root.title("AirInk")
        root.configure(bg="#f3f5f9")
        root.resizable(False, False)
        self.build_ui()
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<Escape>", lambda event: self.close())
        self.tick()

    def build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", font=("Helvetica", 11), padding=(10, 9),
                        background="#edf1f7", foreground="#172033", borderwidth=0, relief="flat")
        style.map("TButton", background=[("active", "#dfe7f4")])
        style.configure("Accent.TButton", background="#2563eb", foreground="white")
        style.map("Accent.TButton", background=[("active", "#1e4aa3")])
        style.configure("TRadiobutton", background="white", font=("Helvetica", 11))
        style.configure("TScale", background="#2563eb", troughcolor="#edf1f7", borderwidth=0)
        header = tk.Frame(self.root, bg="#f3f5f9")
        header.pack(fill="x", padx=24, pady=(20, 16))
        tk.Label(header, text="AirInk", font=("Helvetica", 26, "bold"),
                 bg="#f3f5f9", fg="#172033").pack(side="left")
        tk.Label(header, textvariable=self.status, font=("Helvetica", 11),
                 bg="#e5eaf4", fg="#35527c", padx=14, pady=7).pack(side="right")
        body = tk.Frame(self.root, bg="#f3f5f9")
        body.pack(padx=24)
        self.screen = tk.Label(body, bd=0, bg="white")
        self.screen.pack(side="left")
        self.screen.bind("<Button-1>", self.mouse_draw)
        self.screen.bind("<B1-Motion>", self.mouse_draw)
        self.screen.bind("<ButtonRelease-1>", lambda event: self.reset_stroke())
        panel = tk.Frame(body, bg="white", width=196, height=HEIGHT)
        panel.pack(side="left", fill="y", padx=(16, 0))
        panel.pack_propagate(False)

        def title(text):
            tk.Label(panel, text=text, bg="white", fg="#67758a",
                     font=("Helvetica", 10, "bold"), anchor="w").pack(
                         fill="x", padx=18, pady=(12, 6))

        self.camera_button = ttk.Button(panel, text="Start camera", style="Accent.TButton",
                                        command=self.toggle_camera)
        self.camera_button.pack(fill="x", padx=16, pady=(18, 0))
        title("TOOL")
        row = tk.Frame(panel, bg="white")
        row.pack(fill="x", padx=16)
        for name in ("Pen", "Eraser"):
            ttk.Radiobutton(row, text=name, value=name, variable=self.tool,
                            command=self.reset_stroke).pack(side="left")
        title("COLOUR")
        row = tk.Frame(panel, bg="white")
        row.pack(padx=16)
        self.swatches = {}
        for index, color in enumerate(COLORS):
            swatch = tk.Label(row, bg=color, width=2, height=1, bd=3, font=("Helvetica", 10),
                              relief="solid" if color == self.color else "flat")
            swatch.grid(row=0, column=index, padx=2, pady=4)
            swatch.bind("<Button-1>", lambda event, c=color: self.choose_color(c))
            self.swatches[color] = swatch
        title("BRUSH SIZE")
        ttk.Scale(panel, from_=2, to=24, variable=self.size,
                  command=lambda value: self.reset_stroke()).pack(fill="x", padx=18)
        title("VIEW")
        row = tk.Frame(panel, bg="white")
        row.pack(fill="x", padx=16)
        for name in ("Camera", "Board"):
            ttk.Radiobutton(row, text=name, value=name, variable=self.view).pack(side="left")
        title("CAMERA ZOOM")
        ttk.Scale(panel, from_=1, to=2.5, variable=self.zoom,
                  command=lambda value: self.reset_stroke()).pack(fill="x", padx=18)
        title("CAMERA OPACITY")
        ttk.Scale(panel, from_=0, to=100, variable=self.opacity).pack(fill="x", padx=18, pady=(0, 10))
        row = tk.Frame(panel, bg="white")
        row.pack(side="bottom", fill="x", padx=16, pady=(0, 18))
        ttk.Button(row, text="Open", width=5, command=self.open_drawing).pack(
            side="left", expand=True, fill="x")
        ttk.Button(row, text="Save", width=5, style="Accent.TButton", command=self.save).pack(
            side="left", expand=True, fill="x", padx=(6, 0))
        ttk.Button(panel, text="Clear", command=self.clear).pack(
            side="bottom", fill="x", padx=16, pady=(0, 8))
        tk.Label(self.root, text="Only index up: draw     •     More fingers: pause     •     Camera off: use mouse",
                 font=("Helvetica", 11), bg="#f3f5f9", fg="#67758a").pack(pady=16)

    def reset_stroke(self):
        self.previous = None

    def choose_color(self, color):
        self.color = color
        self.tool.set("Pen")
        self.reset_stroke()
        for value, swatch in self.swatches.items():
            swatch.configure(relief="solid" if value == color else "flat")

    def toggle_camera(self):
        if self.camera is not None:
            self.stop_camera()
            return
        camera, tracker = None, None
        try:
            options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=str(ROOT / "assets" / "hand_landmarker.task")),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_hands=1, min_hand_detection_confidence=0.6,
                min_hand_presence_confidence=0.6, min_tracking_confidence=0.6)
            tracker = mp.tasks.vision.HandLandmarker.create_from_options(options)
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
        self.pose_frames = 0
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
        self.reset_stroke()
        self.camera_button.configure(text="Start camera")
        self.status.set("Camera off")

    def draw(self, point):
        point = tuple(int(np.clip(v, 0, limit - 1))
                      for v, limit in zip(point, (WIDTH, HEIGHT)))
        if self.previous is not None:
            if np.linalg.norm(np.subtract(point, self.previous)) > 120:
                self.reset_stroke()
            else:
                point = tuple(int(0.5 * a + 0.5 * b) for a, b in zip(point, self.previous))
        start = self.previous or point
        erase = self.tool.get() == "Eraser"
        thickness = max(24, self.size.get() * 4) if erase else max(2, self.size.get())
        color = tuple(int(self.color[i:i + 2], 16) for i in (1, 3, 5))
        cv2.line(self.ink, start, point, (0, 0, 0) if erase else color, thickness)
        cv2.line(self.mask, start, point, 0 if erase else 255, thickness)
        self.previous = point

    def mouse_draw(self, event):
        if self.camera is None:
            self.draw((event.x, event.y))

    def compose(self, background=None):
        if background is None:
            background = np.full_like(self.ink, 255)
        else:
            opacity = self.opacity.get() / 100
            background = cv2.addWeighted(background, opacity, np.full_like(self.ink, 255), 1 - opacity, 0)
        return np.where(self.mask[:, :, None] > 0, self.ink, background)

    def camera_view(self, frame):
        """Crop only the display; the tracker receives the full camera frame."""
        height, width = frame.shape[:2]
        crop_w, crop_h = round(width / self.zoom.get()), round(height / self.zoom.get())
        left, top = (width - crop_w) // 2, (height - crop_h) // 2
        self.crop = (left / width, top / height, crop_w / width, crop_h / height)
        return cv2.resize(frame[top:top + crop_h, left:left + crop_w], (WIDTH, HEIGHT))

    def view_point(self, tip):
        left, top, width, height = self.crop
        x, y = (tip.x - left) / width, (tip.y - top) / height
        if 0 <= x <= 1 and 0 <= y <= 1:
            return int(x * (WIDTH - 1)), int(y * (HEIGHT - 1))
        return None

    def tick(self):
        background = None
        point = None
        if self.camera is not None:
            try:
                ok, frame = self.camera.read()
                if not ok:
                    raise RuntimeError("Camera disconnected. You can continue with the mouse.")
                frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                self.timestamp = max(self.timestamp + 1, time.monotonic_ns() // 1_000_000)
                result = self.tracker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=frame), self.timestamp)
                preview = self.camera_view(frame)
                hand = result.hand_landmarks[0] if result.hand_landmarks else None
                point = self.view_point(hand[8]) if hand else None
                writing = point is not None and writing_pose(hand)
                self.pose_frames = self.pose_frames + 1 if writing else 0
                self.status.set(("Drawing" if self.tool.get() == "Pen" else "Erasing")
                                if self.pose_frames >= 3 else
                                ("Move to the centre" if hand and point is None else
                                 "Move / pause" if hand else "Show your hand"))
                if self.pose_frames >= 3:
                    self.draw(point)
                else:
                    self.reset_stroke()
                if self.view.get() == "Camera":
                    background = preview
            except Exception as error:
                self.stop_camera()
                messagebox.showerror("Camera stopped", str(error))
        picture = self.compose(background)
        if self.camera is not None and point is not None:
            cv2.circle(picture, point, 10, (22, 155, 131), 2)
        self.photo = ImageTk.PhotoImage(Image.fromarray(picture))
        self.screen.configure(image=self.photo)
        self.after_id = self.root.after(20, self.tick)

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
            path = filedialog.askopenfilename(
                title="Open saved drawing", initialdir=str(folder),
                filetypes=[("PNG drawings", "*.png")])
            if not path:
                return
            with Image.open(path) as image:
                if image.format != "PNG" or image.size != (WIDTH, HEIGHT):
                    raise ValueError("Choose a 960 × 540 PNG saved by AirInk.")
                rgba = image.convert("RGBA")
                white = Image.new("RGBA", image.size, "white")
                ink = np.array(Image.alpha_composite(white, rgba).convert("RGB"))
            self.ink = ink
            self.mask = np.any(ink != 255, axis=2).astype(np.uint8) * 255
            self.pose_frames = 0
            self.reset_stroke()
            self.root.title(f"AirInk — {Path(path).name}")
        except (OSError, ValueError) as error:
            messagebox.showerror("Could not open", str(error))

    def close(self):
        self.root.after_cancel(self.after_id)
        self.stop_camera()
        self.root.destroy()


if __name__ == "__main__":
    AirInk(tk.Tk()).root.mainloop()
