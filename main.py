"""AirInk: raise only your index finger to write."""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, simpledialog, ttk

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk

import art
import db
from art import HEIGHT, WIDTH

ROOT = Path(__file__).resolve().parent
FONT = "Segoe UI" if sys.platform == "win32" else "Helvetica"
PAD, GAP, LEFT, RIGHT = 18, 12, 136, 236
TOTAL = 2 * PAD + LEFT + 2 * GAP + WIDTH + RIGHT
UNDO_LIMIT, UNDO_HOLD = 30, 1.2  # undo steps kept; seconds to hold ✌ for undo
SESSION_COUNTS = ("strokes", "shapes", "flowers", "erases", "undos", "saves")  # columns of the sessions table

PALETTE = ("#2563eb", "#06b6d4", "#169b83", "#22c55e", "#facc15", "#f97316",
           "#e34b65", "#ec4899", "#8b5cf6", "#92400e", "#172033", "#ffffff")
TOOLS = (("Draw", "✎"), ("Eraser", "⌫"), ("Shapes", "◇"), ("Flowers", "✿"))
BRUSHES = {  # name: (button label, opacity)
    "Pen": ("✎ Pen", 1.0), "Marker": ("▮ Marker", 0.9), "Highlighter": ("▬ Highlighter", 0.38),
    "Neon": ("✦ Neon glow", 1.0), "Rainbow": ("◐ Rainbow", 1.0), "Dotted": ("⋯ Dotted", 1.0),
    "Spray": ("⁂ Spray", 1.0), "Calligraphy": ("✒ Calligraphy", 1.0), "Chalk": ("▒ Chalk", 0.95)}
SHAPE_ICONS = dict(zip(art.SHAPES, "─→▭◯△☆♡◇"))
FLOWER_ICONS = dict(zip(art.FLOWERS, "✿❀⚘❁✾❦❋❧"))
STATUS_COLORS = {"draw": "#16a34a", "pause": "#d97706", "idle": "#64748b", "undo": "#9333ea"}
THEMES = {
    "Pastel Dream": dict(bg="#f5f0ff", panel="#ffffff", text="#3b2a5a", muted="#8b7bab", accent="#8b5cf6",
                         hover="#ece3ff", button="#f4eeff", grad=("#c4b5fd", "#f9a8d4", "#93c5fd")),
    "Neon Night": dict(bg="#0b0b16", panel="#16162b", text="#ecebff", muted="#8f8cc0", accent="#a855f7",
                       hover="#302c5c", button="#222041", grad=("#6d28d9", "#a855f7", "#06b6d4")),
    "Sunny Vibes": dict(bg="#fff7ed", panel="#ffffff", text="#4a2c0a", muted="#b07a3c", accent="#f97316",
                        hover="#ffe2c2", button="#fff1e0", grad=("#fb923c", "#facc15", "#2dd4bf")),
    "Ocean Glass": dict(bg="#e6f4ff", panel="#f8fbff", text="#0c3b5e", muted="#5a87a8", accent="#0ea5e9",
                        hover="#d3eaff", button="#eaf5ff", grad=("#38bdf8", "#818cf8", "#c084fc")),
    "Classic": dict(bg="#f3f5f9", panel="#ffffff", text="#172033", muted="#67758a", accent="#2563eb",
                    hover="#dfe7f4", button="#edf1f7", grad=("#1d4ed8", "#2563eb", "#60a5fa")),
}


def app_folder():
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path.home() / ".local" / "share"
    return base / "AirInk"


def storage_folder():
    return app_folder() / "drawings"


def gesture(hand):
    """Name the hand pose: point (only index up), peace, palm, fist or other."""
    raised = [hand[tip].y < hand[tip - 2].y - 0.025 for tip in (8, 12, 16, 20)]

    def distance(a, b):
        return np.hypot(hand[a].x - hand[b].x, hand[a].y - hand[b].y)

    thumb_open = distance(4, 17) > 1.15 * distance(3, 17)
    if raised == [True, False, False, False]:
        return "other" if thumb_open else "point"
    if raised == [True, True, False, False]:
        return "peace"
    if all(raised):
        return "palm"
    return "other" if any(raised) else "fist"


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
        self.smoother, self.color, self.rng = OneEuro(), PALETTE[0], np.random.default_rng()
        self.ink = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)   # colour of each pixel
        self.mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)     # 0 = empty, 255 = ink
        self.layer = art.Layer()                                  # the stroke in progress
        self.painter = art.Painter(self.layer)
        self.undo_stack, self.redo_stack = [], []
        self.active, self.hover, self.last_frame = None, None, None
        self.travel = self.carry = self.hue0 = 0.0
        self.shape_start = self.shape_end = self.last_stamp = None
        self.peace_since, self.peace_missed, self.peace_done = None, 0, False
        self.crop, self.toast_id, self.boards = (0, 0, 1, 1), None, {}
        self.status_state = ("Camera off · draw with the mouse", "idle")
        self.session_id, self.session_start = None, datetime.now().replace(microsecond=0)
        self.stats = dict.fromkeys(SESSION_COUNTS, 0)
        self.used_camera = False
        try:
            db.init_db()
            self.db_error = None
        except db.Error as error:
            self.db_error = str(error)

        settings = self.load_settings()
        self.theme = tk.StringVar(value=settings.get("theme") if settings.get("theme") in THEMES else "Pastel Dream")
        self.board = tk.StringVar(value=settings.get("board") if settings.get("board") in art.BOARDS else "White")
        self.tool, self.brush = tk.StringVar(value="Draw"), tk.StringVar(value="Pen")
        self.shape, self.flower = tk.StringVar(value="Rectangle"), tk.StringVar(value="Daisy")
        self.fill, self.natural = tk.BooleanVar(value=False), tk.BooleanVar(value=True)
        self.stem, self.trail = tk.BooleanVar(value=True), tk.BooleanVar(value=False)
        self.view = tk.StringVar(value="Camera")
        self.size, self.alpha, self.opacity = tk.IntVar(value=6), tk.IntVar(value=100), tk.IntVar(value=60)
        self.status, self.toast_text, self.hex = tk.StringVar(), tk.StringVar(), tk.StringVar(value=self.color)

        root.title("AirInk")
        root.resizable(False, False)
        self.build_ui()
        mod = "Command" if sys.platform == "darwin" else "Control"
        for keys, action in (((f"<{mod}-z>",), self.undo), ((f"<{mod}-Z>", f"<{mod}-y>"), self.redo),
                             ((f"<{mod}-s>",), self.save), (("<bracketleft>",), lambda: self.nudge_size(-2)),
                             (("<bracketright>",), lambda: self.nudge_size(2))):
            for key in keys:
                root.bind(key, lambda event, action=action: action())
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<Escape>", lambda event: self.close())
        if self.db_error:
            self.toast("MySQL not connected: saving files only")
        self.tick()

    # ---------- settings ----------
    @staticmethod
    def load_settings():
        try:
            return json.loads((app_folder() / "settings.json").read_text())
        except (OSError, ValueError):
            return {}

    def save_settings(self):
        try:
            app_folder().mkdir(parents=True, exist_ok=True)
            (app_folder() / "settings.json").write_text(json.dumps(
                {"theme": self.theme.get(), "board": self.board.get()}))
        except OSError:
            pass

    # ---------- interface ----------
    def build_ui(self):
        for child in self.root.winfo_children():
            child.destroy()
        t = self.t = THEMES[self.theme.get()]
        self.chips, self.swatches = [], {}
        self.root.configure(bg=t["bg"])
        for option, value in (("background", t["panel"]), ("foreground", t["text"]),
                              ("selectBackground", t["accent"]), ("selectForeground", "white")):
            self.root.option_add(f"*TCombobox*Listbox.{option}", value)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Horizontal.TScale", background=t["accent"], troughcolor=t["button"],
                        bordercolor=t["panel"], lightcolor=t["accent"], darkcolor=t["accent"])
        style.configure("TCombobox", fieldbackground=t["panel"], background=t["button"], foreground=t["text"],
                        arrowcolor=t["text"], bordercolor=t["panel"], lightcolor=t["panel"], darkcolor=t["panel"])
        style.map("TCombobox", fieldbackground=[("readonly", t["panel"])], foreground=[("readonly", t["text"])],
                  selectbackground=[("readonly", t["panel"])], selectforeground=[("readonly", t["text"])])

        self.build_header()
        body = tk.Frame(self.root, bg=t["bg"])
        body.pack(padx=PAD)
        left = tk.Frame(body, bg=t["panel"], width=LEFT, height=HEIGHT)
        left.pack(side="left")
        left.pack_propagate(False)
        self.screen = tk.Label(body, bd=0, bg=t["panel"], cursor="none" if sys.platform == "darwin" else "crosshair")
        self.screen.pack(side="left", padx=GAP)
        self.screen.bind("<ButtonPress-1>", lambda e: self.camera is None and self.begin((e.x, e.y)))
        self.screen.bind("<B1-Motion>", self.mouse_move)
        self.screen.bind("<ButtonRelease-1>", lambda e: self.camera is None and self.end())
        self.screen.bind("<Motion>", self.mouse_move)
        self.screen.bind("<Leave>", lambda e: setattr(self, "hover", None))
        right = tk.Frame(body, bg=t["panel"], width=RIGHT, height=HEIGHT)
        right.pack(side="left")
        right.pack_propagate(False)
        self.build_left(left)
        self.build_right(right)

        bottom = tk.Frame(self.root, bg=t["bg"])
        bottom.pack(fill="x", padx=PAD, pady=(GAP, PAD))
        tk.Label(bottom, text="☝ Index up: draw   ✋ Open hand: pause   ✌ Hold two: undo",
                 font=(FONT, 11), bg=t["bg"], fg=t["muted"]).pack(side="left")
        tk.Label(bottom, textvariable=self.toast_text, font=(FONT, 11, "bold"), width=44, anchor="w",
                 bg=t["bg"], fg=t["accent"]).pack(side="left", padx=12)
        for text, command, accent in (("⬇ Save PNG", self.save, True), ("◉ Snapshot", self.snapshot, False),
                                      ("▦ Gallery", self.gallery, False), ("▤ My stats", self.show_stats, False),
                                      ("Open…", self.open_drawing, False)):
            self.chip(bottom, text, command, accent=accent).pack(side="right", padx=(6, 0))
        self.set_status(*self.status_state)
        self.refresh_chips()

    def build_header(self):
        t, width, height = self.t, TOTAL - 2 * PAD, 64
        canvas = tk.Canvas(self.root, width=width, height=height, bg=t["bg"], highlightthickness=0, bd=0)
        canvas.pack(padx=PAD, pady=(PAD, GAP))
        self.banner = ImageTk.PhotoImage(art.banner(width, height, t["grad"], t["bg"]))
        canvas.create_image(0, 0, image=self.banner, anchor="nw")
        canvas.create_text(24, height / 2, text="✿ AirInk", font=(FONT, 24, "bold"), fill="white", anchor="w")
        canvas.create_text(168, height / 2 + 2, text="draw in the air with your finger",
                           font=(FONT, 12), fill="white", anchor="w")
        x = 470
        for label, variable, values, command in (("Theme", self.theme, list(THEMES), self.change_theme),
                                                 ("Board", self.board, art.BOARDS, self.change_board)):
            canvas.create_text(x, height / 2, text=label, font=(FONT, 12, "bold"), fill="white", anchor="w")
            box = ttk.Combobox(canvas, textvariable=variable, values=values, state="readonly",
                               width=13, font=(FONT, 11))
            box.bind("<<ComboboxSelected>>", lambda event, command=command: command())
            canvas.create_window(x + 56, height / 2, window=box, anchor="w")
            x += 240
        self.status_pill = tk.Label(canvas, textvariable=self.status, font=(FONT, 12, "bold"),
                                    fg="white", bg=STATUS_COLORS["idle"], padx=14, pady=6)
        canvas.create_window(width - 18, height / 2, window=self.status_pill, anchor="e")
        self.status_kind = None

    def build_left(self, panel):
        self.camera_button = self.chip(panel, "■ Stop camera" if self.camera else "● Start camera",
                                       self.toggle_camera, accent=True)
        self.camera_button.pack(fill="x", padx=12, pady=(14, 0))
        self.title(panel, "TOOLS")
        for name, icon in TOOLS:
            self.choice(panel, f"{icon}  {name}", self.tool, name, self.change_tool).pack(fill="x", padx=12, pady=2)
        self.title(panel, "EDIT")
        for text, command in (("↶  Undo", self.undo), ("↷  Redo", self.redo), ("✕  Clear", self.clear)):
            self.chip(panel, text, command, anchor="w").pack(fill="x", padx=12, pady=2)
        self.title(panel, "CAMERA VIEW")
        row = tk.Frame(panel, bg=self.t["panel"])
        row.pack(fill="x", padx=12)
        for name in ("Camera", "Board"):
            self.choice(row, name, self.view, name, size=10, padx=4).pack(side="left", expand=True, fill="x", padx=1)
        self.title(panel, "CAMERA FADE")
        ttk.Scale(panel, from_=0, to=100, variable=self.opacity).pack(fill="x", padx=14)

    def build_right(self, panel):
        t = self.t
        self.options = tk.Frame(panel, bg=t["panel"])
        self.options.pack(fill="x")
        self.show_options()
        self.title(panel, "COLOUR")
        grid = tk.Frame(panel, bg=t["panel"])
        grid.pack(padx=12, anchor="w")
        for index, color in enumerate(PALETTE):
            swatch = tk.Label(grid, bg=color, width=2, height=1, cursor="hand2",
                              highlightthickness=3, highlightbackground=t["panel"])
            swatch.grid(row=index // 6, column=index % 6, padx=2, pady=2)
            swatch.bind("<Button-1>", lambda event, c=color: self.choose_color(c))
            self.swatches[color] = swatch
        row = tk.Frame(panel, bg=t["panel"])
        row.pack(fill="x", padx=14, pady=(6, 0))
        self.preview = tk.Label(row, bg=self.color, width=3, highlightthickness=1, highlightbackground=t["muted"])
        self.preview.pack(side="left")
        entry = tk.Entry(row, textvariable=self.hex, width=8, font=(FONT, 11), relief="flat", bg=t["button"],
                         fg=t["text"], insertbackground=t["text"], highlightthickness=0)
        entry.pack(side="left", padx=6, ipady=4)
        entry.bind("<Return>", lambda event: self.typed_color())
        self.chip(row, "◍ Custom", self.pick_color, size=10).pack(side="left", fill="x", expand=True)
        self.title(panel, "SIZE   ( [ and ] keys )")
        ttk.Scale(panel, from_=2, to=24, variable=self.size).pack(fill="x", padx=16)
        self.title(panel, "INK OPACITY")
        ttk.Scale(panel, from_=10, to=100, variable=self.alpha).pack(fill="x", padx=16)
        self.mark_color()

    def show_options(self):
        """The top of the right panel changes with the tool: brushes, shapes or flowers."""
        for child in self.options.winfo_children():
            child.destroy()
        self.chips = [chip for chip in self.chips if chip[2].winfo_exists()]
        tool, t = self.tool.get(), self.t
        if tool == "Eraser":
            self.title(self.options, "ERASER")
            tk.Label(self.options, text="Rub with your finger (or drag the mouse) to erase. "
                     "SIZE sets the eraser width.", wraplength=RIGHT - 32, justify="left",
                     font=(FONT, 11), bg=t["panel"], fg=t["text"]).pack(fill="x", padx=14)
            return
        title, variable, names = {"Draw": ("BRUSH", self.brush, BRUSHES),
                                  "Shapes": ("SHAPE", self.shape, art.SHAPES),
                                  "Flowers": ("FLOWER STAMP", self.flower, art.FLOWERS)}[tool]
        self.title(self.options, title)
        grid = tk.Frame(self.options, bg=t["panel"])
        grid.pack(fill="x", padx=10)
        grid.columnconfigure((0, 1), weight=1, uniform="column")
        for index, name in enumerate(names):
            label = (BRUSHES[name][0] if tool == "Draw" else
                     f"{(SHAPE_ICONS if tool == 'Shapes' else FLOWER_ICONS)[name]} {name}")
            self.choice(grid, label, variable, name, self.end, size=10, padx=6).grid(
                row=index // 2, column=index % 2, sticky="ew", padx=2, pady=2)
        toggles = {"Shapes": ((self.fill, "Fill"),),
                   "Flowers": ((self.natural, "Natural"), (self.stem, "Stem"), (self.trail, "Scatter"))}.get(tool, ())
        row = tk.Frame(self.options, bg=t["panel"])
        row.pack(fill="x", padx=10, pady=(4, 0))
        row.columnconfigure((0, 1), weight=1, uniform="column")
        for index, (variable, text) in enumerate(toggles):
            self.toggle(row, text, variable).grid(row=index // 2, column=index % 2, sticky="ew", padx=2, pady=2)

    def title(self, parent, text):
        tk.Label(parent, text=text, bg=self.t["panel"], fg=self.t["muted"], font=(FONT, 9, "bold"),
                 anchor="w").pack(fill="x", padx=14, pady=(12, 4))

    def chip(self, parent, text, command, accent=False, size=11, anchor="center", padx=10):
        """A flat, colourful button that looks the same on every platform."""
        t = self.t
        colors = ("white", t["accent"]) if accent else (t["text"], t["button"])
        label = tk.Label(parent, text=text, fg=colors[0], bg=colors[1], padx=padx, pady=5, anchor=anchor,
                         font=(FONT, size, "bold" if accent else "normal"), cursor="hand2")
        label.colors = colors
        label.bind("<Button-1>", lambda event: command())
        label.bind("<Enter>", lambda event: label.colors[1] == t["button"] and label.configure(bg=t["hover"]))
        label.bind("<Leave>", lambda event: label.configure(bg=label.colors[1]))
        return label

    def choice(self, parent, text, variable, value, command=None, size=11, padx=10):
        def pick():
            variable.set(value)
            self.refresh_chips()
            if command:
                command()
        label = self.chip(parent, text, pick, size=size, anchor="w", padx=padx)
        self.chips.append((variable, value, label))
        self.refresh_chips()
        return label

    def toggle(self, parent, text, variable):
        def flip():
            variable.set(not variable.get())
            self.refresh_chips()
        label = self.chip(parent, text, flip, size=10, padx=6)
        self.chips.append((variable, True, label))
        self.refresh_chips()
        return label

    def refresh_chips(self):
        t = self.t
        for variable, value, label in self.chips:
            if not label.winfo_exists():
                continue
            on = variable.get() == value
            if isinstance(variable, tk.BooleanVar):
                label.configure(text=("☑ " if on else "☐ ") + label.cget("text").lstrip("☑☐ "))
            label.colors = ("white", t["accent"]) if on else (t["text"], t["button"])
            label.configure(fg=label.colors[0], bg=label.colors[1])

    def set_status(self, text, kind):
        self.status_state = (text, kind)
        if self.status.get() != text:
            self.status.set(text)
        if self.status_kind != kind:
            self.status_pill.configure(bg=STATUS_COLORS[kind])
            self.status_kind = kind

    def toast(self, text):
        self.toast_text.set(text)
        if self.toast_id:
            self.root.after_cancel(self.toast_id)
        self.toast_id = self.root.after(2500, lambda: self.toast_text.set(""))

    def change_theme(self):
        self.save_settings()
        self.root.after_idle(self.build_ui)  # don't destroy the combobox inside its own event

    def change_board(self):
        self.save_settings()
        self.root.focus_set()

    def change_tool(self):
        self.end()
        self.show_options()

    def nudge_size(self, step):
        self.size.set(int(np.clip(self.size.get() + step, 2, 24)))

    # ---------- colour ----------
    def choose_color(self, color):
        self.end()
        self.color = color.lower()
        self.hex.set(self.color)
        if self.tool.get() == "Eraser":
            self.tool.set("Draw")
            self.refresh_chips()
            self.show_options()
        self.mark_color()

    def mark_color(self):
        for value, swatch in self.swatches.items():
            swatch.configure(highlightbackground=self.t["text"] if value == self.color else self.t["panel"])
        self.preview.configure(bg=self.color)

    def pick_color(self):
        chosen = colorchooser.askcolor(color=self.color, parent=self.root, title="Choose ink colour")[1]
        if chosen:
            self.choose_color(chosen)

    def typed_color(self):
        value = self.hex.get().strip().lstrip("#")
        if len(value) == 3:
            value = "".join(ch * 2 for ch in value)
        try:
            int(value, 16)
            if len(value) != 6:
                raise ValueError
        except ValueError:
            self.hex.set(self.color)
            return self.toast("Type a colour like #ff66aa")
        self.choose_color("#" + value)
        self.root.focus_set()

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
        self.end()
        self.camera, self.tracker = camera, tracker
        self.used_camera = True
        self.pose_frames = self.failed_reads = 0
        self.camera_button.configure(text="■ Stop camera")
        self.set_status("Show your hand", "idle")

    def stop_camera(self):
        if self.camera is not None:
            self.camera.release()
        if self.tracker is not None:
            self.tracker.close()
        self.camera = self.tracker = self.last_frame = None
        self.pose_frames = 0
        self.peace_since = None
        self.smoother.reset()
        self.end()
        self.camera_button.configure(text="● Start camera")
        self.set_status("Camera off · draw with the mouse", "idle")

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

    # ---------- actions: begin, move, end ----------
    def clip(self, point):
        return np.clip(np.array(point, dtype=float), 0, (WIDTH - 1, HEIGHT - 1))

    def begin(self, point):
        self.end()
        point, tool = self.clip(point), self.tool.get()
        self.active, self.stroke = tool, []
        self.travel, self.carry, self.hue0 = 0.0, np.inf, self.rng.random()
        if tool == "Draw":
            brush = self.brush.get()
            glow = max(4.0, self.size.get() * 1.2) if brush == "Neon" else 0.0
            self.layer.start(BRUSHES[brush][1] * self.alpha.get() / 100, glow)
            self.draw(point)
        elif tool == "Eraser":
            self.push_undo()
            self.draw(point)
        elif tool == "Shapes":
            self.layer.start(self.alpha.get() / 100)
            self.shape_start = point
            self.shape_to(point)
        else:
            self.push_undo()
            self.place_flower(point)

    def move(self, point, camera=False):
        if self.active is None:
            return
        point = self.clip(point)
        if self.active in ("Draw", "Eraser"):
            self.draw(point, camera)
        elif self.active == "Shapes":
            self.shape_to(point)
        elif self.trail.get() and np.linalg.norm(point - self.last_stamp) > self.flower_radius() * 1.8:
            self.place_flower(point)

    def end(self):
        if self.active is None:
            return
        if self.active in ("Draw", "Eraser"):
            self.reset_stroke()
        if self.active == "Shapes" and np.linalg.norm(self.shape_end - self.shape_start) < 4:
            self.layer.clear()  # a click, not a drag: no shape
        if self.active == "Eraser":
            self.stats["erases"] += 1
        if not self.layer.empty:
            self.stats["shapes" if self.active == "Shapes" else "strokes"] += 1
            self.push_undo()
            self.layer.merge(self.ink, self.mask)
            self.layer.clear()
        self.active = None

    def mouse_move(self, event):
        self.hover = (event.x, event.y)
        if self.camera is None and self.active is not None:
            self.move((event.x, event.y))

    # ---------- undo ----------
    def push_undo(self):
        self.undo_stack.append((self.ink.copy(), self.mask.copy()))
        del self.undo_stack[:-UNDO_LIMIT]
        self.redo_stack.clear()

    def undo(self):
        self.end()
        if not self.undo_stack:
            return self.toast("Nothing to undo")
        self.redo_stack.append((self.ink, self.mask))
        self.ink, self.mask = self.undo_stack.pop()
        self.stats["undos"] += 1
        self.toast(f"↶ Undone  ({len(self.undo_stack)} more)")

    def redo(self):
        self.end()
        if not self.redo_stack:
            return self.toast("Nothing to redo")
        self.undo_stack.append((self.ink, self.mask))
        self.ink, self.mask = self.redo_stack.pop()
        self.toast("↷ Redone")

    # ---------- drawing ----------
    def reset_stroke(self):
        """End the curve: finish the last half-segment, then start fresh."""
        if len(self.stroke) >= 2:
            self.paint([(self.stroke[-2] + self.stroke[-1]) / 2, self.stroke[-1]])
        self.stroke = []

    def draw(self, point, camera=False):
        """Add a point to the stroke and draw a smooth curve through the last three points."""
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

    def walk(self, curve, spacing):
        """Points spaced evenly along the stroke, carrying the gap over between calls."""
        points = []
        if self.carry >= spacing:  # the very first point of a stroke
            points.append(curve[0])
            self.carry = 0.0
        for a, b in zip(curve[:-1], curve[1:]):
            length = np.linalg.norm(b - a)
            at = spacing - self.carry
            while at <= length:
                points.append(a + (b - a) * at / length)
                at += spacing
            self.carry = (self.carry + length) % spacing if length else self.carry
        return points

    def paint(self, curve):
        curve = np.asarray(curve, dtype=float)
        size, p = self.size.get(), self.painter
        if self.active == "Eraser":
            cv2.polylines(self.mask, [art.fixed(curve)], False, 0, max(24, size * 4), cv2.LINE_AA, art.SHIFT)
            return
        brush, color = self.brush.get(), art.rgb(self.color)
        if brush == "Pen":
            p.line(curve, color, size)
        elif brush == "Marker":
            p.line(curve, color, size * 2.2)
        elif brush == "Highlighter":
            p.line(curve, color, size * 3.5)
        elif brush == "Neon":
            p.line(curve, color, size, ink_width=size + self.layer.glow * 6)
        elif brush == "Rainbow":
            p.line(curve, art.hue(self.hue0 + self.travel / 400), size * 1.5)
        elif brush == "Calligraphy":
            p.nib(curve, color, size * 2.2)
        elif brush == "Chalk":
            p.chalk(curve, color, size * 1.8)
        elif brush == "Dotted":
            p.dots(self.walk(curve, size * 2.6 + 4), size * 0.8 + 1, color)
        elif brush == "Spray":
            for center in self.walk(curve, 3):
                spread = self.rng.normal(0, size * 1.1, (4 + size // 3, 2))
                p.dots(center + spread, 0.8, color)
        self.travel += np.linalg.norm(np.diff(curve, axis=0), axis=1).sum()

    def shape_to(self, point):
        self.shape_end = point
        self.layer.clear()
        color, size = art.rgb(self.color), self.size.get()
        for points, closed, solid in art.shape_outline(self.shape.get(), self.shape_start, point):
            if solid or (closed and self.fill.get()):
                self.painter.polygon(points, color)
            self.painter.line(points, color, size, closed)

    def flower_radius(self):
        return 14 + self.size.get() * 4

    def place_flower(self, point):
        color = None if self.natural.get() else art.rgb(self.color)
        self.layer.start(self.alpha.get() / 100)
        art.flower(self.painter, self.flower.get(), point, self.flower_radius(), color, self.stem.get(), self.rng)
        self.layer.merge(self.ink, self.mask)
        self.layer.clear()
        self.last_stamp = point
        self.stats["flowers"] += 1

    def board_image(self):
        name = self.board.get()
        if name not in self.boards:
            self.boards[name] = art.board(name)
        return self.boards[name]

    def compose(self, camera_frame=None):
        """Blend ink over the board, or over the camera faded into the board."""
        background = self.board_image()
        if camera_frame is not None:
            opacity = self.opacity.get() / 100
            background = cv2.addWeighted(camera_frame, opacity, background, 1 - opacity, 0)
        ink, mask = self.ink, self.mask
        if not self.layer.empty:
            ink, mask = ink.copy(), mask.copy()
            self.layer.merge(ink, mask)
        alpha = mask[:, :, None].astype(np.uint16)
        return ((ink * alpha + background * (255 - alpha)) // 255).astype(np.uint8)

    def draw_cursor(self, picture, point):
        center, tool, size = tuple(int(v) for v in point), self.tool.get(), self.size.get()
        color = art.rgb(self.color)
        radius = {"Eraser": max(24, size * 4) / 2, "Flowers": self.flower_radius(), "Shapes": 5}.get(
            tool, max(4, size * {"Marker": 2.2, "Highlighter": 3.5, "Spray": 2.2}.get(self.brush.get(), 1) / 2))
        if tool == "Eraser":
            color = (120, 120, 120)
        cv2.circle(picture, center, int(radius) + 2, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.circle(picture, center, int(radius) + 2, color, 2, cv2.LINE_AA)
        if self.peace_since is not None and not self.peace_done:
            progress = min(1.0, (time.monotonic() - self.peace_since) / UNDO_HOLD)
            cv2.ellipse(picture, center, (30, 30), -90, 0, 360 * progress, (147, 51, 234), 5, cv2.LINE_AA)

    # ---------- main loop ----------
    def tick(self):
        point = None
        try:
            if self.camera is not None:
                self.last_frame, point = self.camera_step()
        except Exception as error:
            self.stop_camera()
            messagebox.showerror("Camera stopped", str(error))
        try:
            frame = self.last_frame if self.view.get() == "Camera" else None
            picture = self.compose(frame)
            cursor = point if self.camera is not None else self.hover
            if cursor is not None:
                self.draw_cursor(picture, cursor)
            self.photo = ImageTk.PhotoImage(Image.fromarray(picture))
            self.screen.configure(image=self.photo)
        finally:
            self.after_id = self.root.after(15, self.tick)

    def camera_step(self):
        """Read one frame, track the hand, act on the pose; return (camera picture, fingertip)."""
        ok, frame = self.camera.read()
        if not ok:  # macOS often returns empty frames while the camera warms up
            self.failed_reads += 1
            if self.failed_reads > 100:
                raise RuntimeError("Camera disconnected. You can continue with the mouse.")
            self.set_status("Starting camera…", "idle")
            return self.last_frame, None
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
        pose = gesture(hand) if hand else None
        writing = point is not None and pose == "point"
        # Start after 3 good frames; stop after 3 missed ones, so one bad frame won't split a line.
        self.missed = 0 if writing else self.missed + 1
        if writing:
            self.pose_frames += 1
        elif self.missed >= 3:
            self.pose_frames = 0
        drawing = writing and self.pose_frames >= 3
        if drawing:
            if self.active is None:
                self.begin(point)
            else:
                self.move(point, camera=True)
        elif self.pose_frames < 3:
            self.end()
        self.undo_gesture(pose)
        self.set_status(*self.describe(pose, drawing))
        return preview, point

    def undo_gesture(self, pose):
        """Hold ✌ (index + middle finger) still for a moment to undo once."""
        now = time.monotonic()
        if pose == "peace" and self.active is None:
            self.peace_missed = 0
            if self.peace_since is None:
                self.peace_since = now
            if not self.peace_done and now - self.peace_since >= UNDO_HOLD:
                self.peace_done = True
                self.undo()
        else:
            self.peace_missed += 1
            if self.peace_missed > 4:
                self.peace_since, self.peace_done = None, False

    def describe(self, pose, drawing):
        if pose is None:
            return "Show your hand", "idle"
        if drawing:
            return {"Draw": f"☝ Drawing · {self.brush.get()}", "Eraser": "☝ Erasing",
                    "Shapes": f"☝ Shaping · {self.shape.get()}",
                    "Flowers": f"☝ Planting · {self.flower.get()}"}[self.tool.get()], "draw"
        if pose == "peace":
            return ("✌ Undone!" if self.peace_done else "✌ Hold to undo…"), "undo"
        return {"palm": "✋ Open hand · paused", "fist": "✊ Fist · paused",
                "point": "☝ Getting ready…"}.get(pose, "Paused · raise only your index"), "pause"

    # ---------- files ----------
    def clear(self):
        self.end()
        if not self.mask.any():
            return
        self.push_undo()
        self.ink = np.zeros_like(self.ink)
        self.mask = np.zeros_like(self.mask)
        self.pose_frames = 0
        self.toast("Cleared — Undo brings it back")

    def save(self):
        self.end()
        try:
            folder = storage_folder()
            (folder / ".layers").mkdir(parents=True, exist_ok=True)
            name = f"AirInk-{datetime.now():%Y%m%d-%H%M%S-%f}.png"
            Image.fromarray(self.compose()).save(folder / name, format="PNG")
            # A transparent copy of just the ink, so Open can keep editing on any board.
            Image.fromarray(np.dstack([self.ink, self.mask])).save(folder / ".layers" / name, format="PNG")
            self.stats["saves"] += 1
        except OSError as error:
            return messagebox.showerror("Could not save", str(error))
        title = f"Drawing {datetime.now():%d %b %H:%M}"
        try:
            if not self.database_ready():
                raise db.Error(self.db_error)
            db.run("INSERT INTO drawings (title, file_name, theme, board) VALUES (%s, %s, %s, %s)",
                   (title, name, self.theme.get(), self.board.get()))
            self.record_session()
            self.toast(f"Saved “{title}”")
        except db.Error:
            self.toast("Saved as a file (MySQL not connected)")

    def snapshot(self):
        """Save exactly what is on screen, camera included."""
        try:
            folder = app_folder() / "snapshots"
            folder.mkdir(parents=True, exist_ok=True)
            frame = self.last_frame if self.camera is not None and self.view.get() == "Camera" else None
            path = folder / f"AirInk-Snap-{datetime.now():%Y%m%d-%H%M%S-%f}.png"
            Image.fromarray(self.compose(frame)).save(path, format="PNG")
            self.toast(f"Snapshot saved to {folder.name}/")
        except OSError as error:
            messagebox.showerror("Could not save snapshot", str(error))

    def open_drawing(self):
        folder = storage_folder()
        folder.mkdir(parents=True, exist_ok=True)
        path = filedialog.askopenfilename(title="Open saved drawing", initialdir=str(folder),
                                          filetypes=[("PNG drawings", "*.png")])
        if path:
            self.load_drawing(Path(path))

    def load_drawing(self, path):
        try:
            layer = path.parent / ".layers" / path.name
            source = layer if layer.exists() else path
            with Image.open(source) as image:
                if image.format != "PNG" or image.size != (WIDTH, HEIGHT):
                    raise ValueError("Choose a 960 × 540 PNG saved by AirInk.")
                if source == layer:
                    rgba = np.array(image.convert("RGBA"))
                    ink, mask = rgba[..., :3].copy(), rgba[..., 3].copy()
                else:  # older saves: ink on white
                    white = Image.new("RGBA", image.size, "white")
                    ink = np.array(Image.alpha_composite(white, image.convert("RGBA")).convert("RGB"))
                    mask = np.any(ink != 255, axis=2).astype(np.uint8) * 255
            self.end()
            self.push_undo()
            self.ink, self.mask = ink, mask
            self.pose_frames = 0
            self.root.title(f"AirInk — {path.name}")
            self.toast(f"Opened {path.name}")
        except (OSError, ValueError) as error:
            messagebox.showerror("Could not open", str(error))

    # ---------- MySQL: gallery and stats ----------
    def database_ready(self):
        """True when MySQL is reachable; retries if it was down when AirInk started."""
        if self.db_error:
            try:
                db.init_db()
                self.db_error = None
            except db.Error as error:
                self.db_error = str(error)
        return self.db_error is None

    def record_session(self):
        """Write this session's counts to the sessions table (the row appears on the first activity)."""
        if self.session_id is None and not (any(self.stats.values()) or self.used_camera):
            return
        values = [datetime.now()] + [self.stats[key] for key in SESSION_COUNTS] + [self.used_camera]
        if self.session_id is None:
            self.session_id = db.run(
                "INSERT INTO sessions (started_at, ended_at, strokes, shapes, flowers, erases, undos, saves,"
                " used_camera) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)", [self.session_start] + values)
        else:
            db.run("UPDATE sessions SET ended_at=%s, strokes=%s, shapes=%s, flowers=%s, erases=%s, undos=%s,"
                   " saves=%s, used_camera=%s WHERE id=%s", values + [self.session_id])

    def sync_drawings(self):
        """Add rows for PNGs saved while MySQL was off, or by an older AirInk."""
        known = {row[0] for row in db.run("SELECT file_name FROM drawings", fetch=True)}
        for path in sorted(storage_folder().glob("AirInk-*.png")):
            if path.name not in known:
                stamp = datetime.fromtimestamp(path.stat().st_mtime).replace(microsecond=0)
                db.run("INSERT INTO drawings (title, file_name, created_at) VALUES (%s, %s, %s)",
                       (f"Drawing {stamp:%d %b %H:%M}", path.name, stamp))

    def find_drawings(self, text=""):
        """Rows (id, title, theme, board, created_at, file_name) matching a title, theme, board or date."""
        self.sync_drawings()
        like = f"%{text}%"
        rows = db.run("SELECT id, title, theme, board, created_at, file_name FROM drawings"
                      " WHERE title LIKE %s OR theme LIKE %s OR board LIKE %s OR CAST(created_at AS CHAR) LIKE %s"
                      " ORDER BY created_at DESC, id DESC", (like,) * 4, fetch=True)
        return rows, db.run("SELECT COUNT(*) FROM drawings", fetch=True)[0][0]

    def rename_drawing(self, file_name, title):
        db.run("UPDATE drawings SET title=%s WHERE file_name=%s", (title.strip()[:100], file_name))

    def delete_drawing(self, file_name):
        """Remove the row, then the PNG and its editable ink layer."""
        db.run("DELETE FROM drawings WHERE file_name=%s", (file_name,))
        for path in (storage_folder() / file_name, storage_folder() / ".layers" / file_name):
            path.unlink(missing_ok=True)

    def table(self, parent, columns, height):
        """A themed Treeview: columns = [(key, heading, width, anchor)]."""
        t, style = self.t, ttk.Style()
        style.configure("AirInk.Treeview", background=t["panel"], fieldbackground=t["panel"], foreground=t["text"],
                        rowheight=28, font=(FONT, 11), borderwidth=0)
        style.map("AirInk.Treeview", background=[("selected", t["accent"])], foreground=[("selected", "white")])
        style.configure("AirInk.Treeview.Heading", background=t["button"], foreground=t["text"],
                        font=(FONT, 11, "bold"), relief="flat", borderwidth=0)
        style.map("AirInk.Treeview.Heading", background=[("active", t["hover"])])
        frame = tk.Frame(parent, bg=t["panel"])
        tree = ttk.Treeview(frame, columns=[c[0] for c in columns], show="headings", height=height,
                            style="AirInk.Treeview", selectmode="browse")
        for key, heading, width, anchor in columns:
            tree.heading(key, text=heading)
            tree.column(key, width=width, anchor=anchor, stretch=False)
        tree.pack(side="left")
        bar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        bar.pack(side="left", fill="y")
        tree.configure(yscrollcommand=bar.set)
        tree.tag_configure("odd", background=t["bg"])
        return frame, tree

    def gallery(self):
        """Saved drawings from MySQL: search by title or date, preview, open, rename or delete."""
        self.end()
        t, folder = self.t, storage_folder()
        window = tk.Toplevel(self.root, bg=t["bg"], padx=18, pady=16)
        window.title("AirInk Gallery")
        window.resizable(False, False)
        top = tk.Frame(window, bg=t["bg"])
        top.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        tk.Label(top, text="✿ Your drawings", font=(FONT, 18, "bold"), bg=t["bg"], fg=t["text"]).pack(side="left")
        query = tk.StringVar()
        self.chip(top, "Show all", lambda: (query.set(""), load())).pack(side="right", padx=(6, 0))
        self.chip(top, "⌕ Search", lambda: load(), accent=True).pack(side="right", padx=(6, 0))
        entry = tk.Entry(top, textvariable=query, width=22, font=(FONT, 12), relief="flat", bg=t["panel"],
                         fg=t["text"], insertbackground=t["text"], highlightthickness=1,
                         highlightbackground=t["hover"], highlightcolor=t["accent"])
        entry.pack(side="right", ipady=5)
        entry.bind("<Return>", lambda event: load())
        tk.Label(top, text="Title, theme, board or date (2026-11-18)", font=(FONT, 10), bg=t["bg"],
                 fg=t["muted"]).pack(side="right", padx=8)
        frame, tree = self.table(window, (("id", "ID", 48, "center"), ("title", "Title", 230, "w"),
                                          ("theme", "Theme", 118, "w"), ("board", "Board", 110, "w"),
                                          ("saved", "Saved", 160, "center")), height=12)
        frame.grid(row=1, column=0, sticky="n")
        side = tk.Frame(window, bg=t["panel"], padx=14, pady=14)
        side.grid(row=1, column=1, sticky="ns", padx=(14, 0))
        preview = tk.Label(side, bg=t["panel"], fg=t["muted"], font=(FONT, 11))
        preview.pack()
        name = tk.Label(side, bg=t["panel"], fg=t["text"], font=(FONT, 13, "bold"), wraplength=300)
        name.pack(pady=(8, 10))
        info = tk.StringVar()
        tk.Label(window, textvariable=info, font=(FONT, 10), bg=t["bg"], fg=t["muted"]).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(10, 0))
        rows = {}

        def fetch(text):
            if self.database_ready():
                try:
                    found, total = self.find_drawings(text)
                    info.set(f"{len(found)} of {total} drawings · MySQL database “{db.DB_CONFIG['database']}”")
                    return found
                except db.Error as error:
                    self.db_error = str(error)
            info.set("MySQL not connected — showing files from the drawings folder. Rename and delete need MySQL.")
            files = sorted(folder.glob("AirInk-*.png"), reverse=True)
            return [("—", f.stem, None, None, datetime.fromtimestamp(f.stat().st_mtime), f.name)
                    for f in files if text.lower() in f.name.lower()]

        def load():
            tree.delete(*tree.get_children())
            rows.clear()
            for index, row in enumerate(fetch(query.get().strip())):
                rows[row[5]] = row
                tree.insert("", "end", iid=row[5], tags=("odd",) if index % 2 else (),
                            values=(row[0], row[1], row[2] or "—", row[3] or "—", f"{row[4]:%d %b %Y · %H:%M}"))
            if rows:
                tree.selection_set(next(iter(rows)))
            show()

        def selected():
            choice = tree.selection()
            return rows.get(choice[0]) if choice else None

        def show(event=None):
            row = selected()
            path = folder / row[5] if row else None
            if path is not None and path.exists():
                with Image.open(path) as image:
                    image.thumbnail((320, 180))
                    window.photo = ImageTk.PhotoImage(image)
                preview.configure(image=window.photo, text="")
            else:
                window.photo = ImageTk.PhotoImage(Image.new("RGB", (320, 180), t["bg"]))
                preview.configure(image=window.photo, text="File missing" if row else "No drawings yet",
                                  compound="center")
            name.configure(text=row[1] if row else "")

        def open_it(event=None):
            row = selected()
            if row and (folder / row[5]).exists():
                window.destroy()
                self.load_drawing(folder / row[5])

        def rename():
            row = selected()
            if not row or row[0] == "—":
                return
            title = simpledialog.askstring("Rename drawing", "New title:", initialvalue=row[1], parent=window)
            if title and title.strip():
                try:
                    self.rename_drawing(row[5], title)
                except db.Error as error:
                    return messagebox.showerror("Database Error", str(error), parent=window)
                load()
                tree.selection_set(row[5])

        def delete():
            row = selected()
            if not row or row[0] == "—":
                return
            if not messagebox.askyesno("Delete drawing", f"Delete “{row[1]}”? This removes the PNG too.",
                                       parent=window):
                return
            try:
                self.delete_drawing(row[5])
            except db.Error as error:
                return messagebox.showerror("Database Error", str(error), parent=window)
            load()

        for text, command, accent in (("Open", open_it, True), ("✎ Rename", rename, False),
                                      ("✕ Delete", delete, False),
                                      ("Open another file…", lambda: (window.destroy(), self.open_drawing()), False)):
            self.chip(side, text, command, accent=accent).pack(fill="x", pady=3)
        tree.bind("<<TreeviewSelect>>", show)
        tree.bind("<Double-1>", open_it)
        load()
        entry.focus_set()

    def show_stats(self):
        """Totals, a 7-day chart and recent sessions, all read from the sessions table."""
        self.end()
        t = self.t
        if not self.database_ready():
            return messagebox.showinfo("My stats", f"Stats are stored in MySQL, which is not connected.\n\n{self.db_error}")
        try:
            self.record_session()
            totals = db.run("SELECT COUNT(*), COALESCE(SUM(TIMESTAMPDIFF(SECOND, started_at, ended_at)), 0),"
                            " COALESCE(SUM(strokes), 0), COALESCE(SUM(shapes), 0), COALESCE(SUM(flowers), 0),"
                            " COALESCE(SUM(undos), 0) FROM sessions", fetch=True)[0]
            saved = db.run("SELECT COUNT(*) FROM drawings", fetch=True)[0][0]
            recent = db.run("SELECT started_at, TIMESTAMPDIFF(SECOND, started_at, ended_at), strokes, shapes,"
                            " flowers, erases, undos, saves, used_camera FROM sessions"
                            " ORDER BY started_at DESC LIMIT 12", fetch=True)
            days = dict(db.run("SELECT DATE(started_at), SUM(strokes + shapes + flowers) FROM sessions"
                               " WHERE started_at >= CURDATE() - INTERVAL 6 DAY GROUP BY DATE(started_at)",
                               fetch=True))
        except db.Error as error:
            return messagebox.showerror("Database Error", str(error))
        sessions, seconds, strokes, shapes, flowers, undos = (int(v) for v in totals)
        window = tk.Toplevel(self.root, bg=t["bg"], padx=18, pady=16)
        window.title("AirInk — My stats")
        window.resizable(False, False)
        tk.Label(window, text="▤ My drawing stats", font=(FONT, 18, "bold"), bg=t["bg"], fg=t["text"]).grid(
            row=0, column=0, columnspan=2, sticky="w")
        now = ", ".join(f"{self.stats[k]} {k}" for k in ("strokes", "shapes", "flowers", "undos"))
        tk.Label(window, text=f"This session: {now}", font=(FONT, 11), bg=t["bg"], fg=t["muted"]).grid(
            row=0, column=1, sticky="e")
        tiles = tk.Frame(window, bg=t["bg"])
        tiles.grid(row=1, column=0, columnspan=2, sticky="ew", pady=12)
        minutes = round(seconds / 60)
        for index, (value, label) in enumerate(((sessions, "Sessions"),
                                                (f"{minutes // 60} h {minutes % 60} m" if minutes >= 60 else
                                                 f"{minutes} min", "Time drawing"),
                                                (strokes, "Strokes"), (shapes, "Shapes"), (flowers, "Flowers"),
                                                (undos, "Undos"), (saved, "Drawings saved"))):
            tile = tk.Frame(tiles, bg=t["panel"], padx=16, pady=10)
            tile.grid(row=0, column=index, padx=(0 if index == 0 else 8, 0), sticky="nsew")
            tiles.columnconfigure(index, weight=1, uniform="tile")
            tk.Label(tile, text=str(value), font=(FONT, 20, "bold"), bg=t["panel"], fg=t["accent"]).pack()
            tk.Label(tile, text=label, font=(FONT, 10), bg=t["panel"], fg=t["muted"]).pack()
        chart = tk.Canvas(window, width=340, height=300, bg=t["panel"], highlightthickness=0)
        chart.grid(row=2, column=0, sticky="n")
        chart.create_text(16, 18, text="Marks made · last 7 days", anchor="w", font=(FONT, 11, "bold"),
                          fill=t["text"])
        dates = [datetime.now().date().fromordinal(datetime.now().date().toordinal() - k) for k in range(6, -1, -1)]
        values = [int(days.get(d, 0) or 0) for d in dates]
        top = max(values + [1])
        for index, (day, value) in enumerate(zip(dates, values)):
            x = 30 + index * 44
            height = 200 * value / top
            chart.create_rectangle(x, 260 - height, x + 28, 260, fill=t["accent"], width=0)
            chart.create_text(x + 14, 252 - height, text=str(value), anchor="s", font=(FONT, 10), fill=t["text"])
            chart.create_text(x + 14, 278, text=f"{day:%a}", font=(FONT, 10), fill=t["muted"])
        frame, tree = self.table(window, (("started", "Started", 150, "center"), ("minutes", "Min", 50, "center"),
                                          ("strokes", "Strokes", 70, "center"), ("shapes", "Shapes", 66, "center"),
                                          ("flowers", "Flowers", 70, "center"), ("erases", "Erases", 62, "center"),
                                          ("undos", "Undos", 60, "center"), ("saves", "Saves", 56, "center"),
                                          ("camera", "Camera", 66, "center")), height=9)
        frame.grid(row=2, column=1, sticky="n", padx=(14, 0))
        for index, row in enumerate(recent):
            started, seconds, *counts, camera = row
            tree.insert("", "end", tags=("odd",) if index % 2 else (),
                        values=(f"{started:%d %b · %H:%M}", max(1, round((seconds or 0) / 60)), *counts,
                                "✓" if camera else "—"))

    def close(self):
        if self.after_id is None:
            return
        self.root.after_cancel(self.after_id)
        self.after_id = None
        self.stop_camera()
        try:
            if self.db_error is None:
                self.record_session()
        except db.Error:
            pass
        self.root.destroy()


if __name__ == "__main__":
    AirInk(tk.Tk()).root.mainloop()
