"""Drawing helpers for AirInk: the stroke layer, brushes, shapes, flower stamps and boards."""

import colorsys
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw

WIDTH, HEIGHT = 960, 540
SHIFT = 4  # 4-bit sub-pixel precision for every OpenCV drawing call

BOARDS = ("White", "Blackboard", "Night sky", "Grid paper", "Dot grid", "Lined paper",
          "Pastel", "Sunset", "Ocean", "Mint")
SHAPES = ("Line", "Arrow", "Rectangle", "Circle", "Triangle", "Star", "Heart", "Diamond")
FLOWERS = ("Daisy", "Sunflower", "Tulip", "Rose", "Blossom", "Lavender", "Butterfly", "Leaf")
GREEN, STEM, BODY = (52, 168, 83), (22, 120, 60), (55, 48, 64)


# ---------- colour ----------
def rgb(hex_color):
    return tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))


def shade(color, factor):
    """factor < 1 darkens towards black, factor > 1 lightens towards white."""
    c = np.array(color, dtype=float)
    c = c * factor if factor <= 1 else c + (255 - c) * min(factor - 1, 1)
    return tuple(int(v) for v in np.clip(c, 0, 255))


def hue(t):
    return tuple(int(v * 255) for v in colorsys.hsv_to_rgb(t % 1, 0.85, 0.95))


def fixed(points):
    return np.round(np.asarray(points, dtype=float).reshape(-1, 2) * (1 << SHIFT)).astype(np.int32)


def ellipse_points(center, axes, angle=0.0, start=0.0, end=360.0, count=None):
    count = count or int(np.clip(max(axes) * 0.9, 18, 120))
    t = np.radians(np.linspace(start, end, count))
    x, y = axes[0] * np.cos(t), axes[1] * np.sin(t)
    a = math.radians(angle)
    return np.stack([center[0] + x * math.cos(a) - y * math.sin(a),
                     center[1] + x * math.sin(a) + y * math.cos(a)], axis=1)


# ---------- the stroke layer ----------
class Layer:
    """A transparent sheet for the stroke in progress, merged onto the drawing when it ends.

    Drawing on a separate sheet keeps translucent brushes even (a highlighter does not
    darken where it crosses itself) and lets a shape be redrawn as it is dragged.
    """

    def __init__(self):
        self.ink = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        self.mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        self.box, self.opacity, self.glow = None, 1.0, 0.0

    @property
    def empty(self):
        return self.box is None

    def start(self, opacity=1.0, glow=0.0):
        self.clear()
        self.opacity, self.glow = opacity, glow

    def clear(self):
        if self.box is not None:
            x0, y0, x1, y1 = self.box
            self.ink[y0:y1, x0:x1] = 0
            self.mask[y0:y1, x0:x1] = 0
        self.box = None

    def touch(self, points, pad):
        """Grow the dirty rectangle to cover these points."""
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        pad += 3 * self.glow + 3
        x0, y0 = np.floor(points.min(0) - pad)
        x1, y1 = np.ceil(points.max(0) + pad) + 1
        box = [max(0, int(x0)), max(0, int(y0)), min(WIDTH, int(x1)), min(HEIGHT, int(y1))]
        if self.box is not None:
            box = [min(box[0], self.box[0]), min(box[1], self.box[1]),
                   max(box[2], self.box[2]), max(box[3], self.box[3])]
        if box[0] < box[2] and box[1] < box[3]:
            self.box = tuple(box)

    def merge(self, ink, mask):
        """Composite this layer over (ink, mask), in place."""
        if self.box is None:
            return
        x0, y0, x1, y1 = self.box
        region = np.s_[y0:y1, x0:x1]
        core = self.mask[region].astype(np.float32) / 255
        color = self.ink[region].astype(np.float32)
        if self.glow:  # neon: a soft halo around a bright, whitish core
            halo = cv2.GaussianBlur(self.mask[region], (0, 0), self.glow).astype(np.float32) / 255
            source = np.maximum(core, np.clip(halo * 2.2, 0, 1) * 0.85)
            k = (core * 0.7)[..., None]
            color = color * (1 - k) + 255 * k
        else:
            source = core
        source = source * self.opacity
        below = mask[region].astype(np.float32) / 255 * (1 - source)
        total = source + below
        mixed = (color * source[..., None] + ink[region] * below[..., None]) / np.maximum(total, 1e-6)[..., None]
        ink[region] = np.clip(mixed + 0.5, 0, 255).astype(np.uint8)
        mask[region] = np.clip(total * 255 + 0.5, 0, 255).astype(np.uint8)


class Painter:
    """Anti-aliased drawing primitives that paint colour and coverage onto a Layer."""

    def __init__(self, layer):
        self.layer = layer
        rng = np.random.default_rng(7)
        grain = np.clip(rng.normal(190, 80, (HEIGHT, WIDTH)), 0, 255).astype(np.uint8)
        self.grain = cv2.GaussianBlur(grain, (0, 0), 0.6)  # chalk texture

    def bleed(self, points, pad, paint):
        """Paint colour a little past a shape's edge so soft edges never show black,
        without spilling onto ink already on the layer."""
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        x0, y0 = np.maximum(np.floor(points.min(0) - pad), 0).astype(int)
        x1, y1 = np.minimum(np.ceil(points.max(0) + pad) + 1, (WIDTH, HEIGHT)).astype(int)
        region = self.layer.ink[y0:y1, x0:x1]
        covered = self.layer.mask[y0:y1, x0:x1] > 0
        kept = region[covered]
        paint()
        region[covered] = kept

    def line(self, points, color, width, closed=False, ink_width=0):
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        width = max(1, int(round(width)))
        ink_width = max(width + 2, int(round(ink_width)))
        line = [fixed(points)]
        self.bleed(points, ink_width, lambda: cv2.polylines(self.layer.ink, line, closed, color, ink_width,
                                                            cv2.LINE_8, SHIFT))
        cv2.polylines(self.layer.ink, line, closed, color, width, cv2.LINE_AA, SHIFT)
        cv2.polylines(self.layer.mask, line, closed, 255, width, cv2.LINE_AA, SHIFT)
        self.layer.touch(points, ink_width / 2 + 2)

    def polygon(self, points, color):
        shape = [fixed(points)]
        self.bleed(points, 3, lambda: cv2.polylines(self.layer.ink, shape, True, color, 3, cv2.LINE_8, SHIFT))
        cv2.fillPoly(self.layer.ink, shape, color, cv2.LINE_AA, SHIFT)
        cv2.fillPoly(self.layer.mask, shape, 255, cv2.LINE_AA, SHIFT)
        self.layer.touch(points, 3)

    def ellipse(self, center, axes, angle, color):
        self.polygon(ellipse_points(center, axes, angle), color)

    def circle(self, center, radius, color):
        self.polygon(ellipse_points(center, (radius, radius)), color)

    def dots(self, points, radius, color):
        r = max(1, int(round(radius * (1 << SHIFT))))
        for point in fixed(points):
            center = tuple(int(v) for v in point)
            cv2.circle(self.layer.ink, center, r + (1 << SHIFT), color, -1, cv2.LINE_8, SHIFT)
            cv2.circle(self.layer.mask, center, r, 255, -1, cv2.LINE_AA, SHIFT)
        if len(points):
            self.layer.touch(points, radius + 2)

    def nib(self, points, color, width, angle=-40):
        """Calligraphy: a flat, angled pen tip — thick one way, thin the other."""
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        n = np.array([math.cos(math.radians(angle)), math.sin(math.radians(angle))]) * width / 2
        for a, b in zip(points[:-1], points[1:]):
            self.polygon([a + n, b + n, b - n, a - n], color)
        self.line(points, color, 2)

    def chalk(self, points, color, width):
        """A grainy line, like chalk dragged over a board."""
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        x0, y0 = np.maximum(np.floor(points.min(0) - width), 0).astype(int)
        x1, y1 = np.minimum(np.ceil(points.max(0) + width) + 1, (WIDTH, HEIGHT)).astype(int)
        if x0 >= x1 or y0 >= y1:
            return
        scratch = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv2.polylines(scratch, [fixed(points - (x0, y0))], False, 255, max(1, int(width)), cv2.LINE_AA, SHIFT)
        region = np.s_[y0:y1, x0:x1]
        self.layer.mask[region] = np.maximum(self.layer.mask[region], np.minimum(scratch, self.grain[region]))
        cv2.polylines(self.layer.ink, [fixed(points)], False, color, int(width) + 2, cv2.LINE_8, SHIFT)
        self.layer.touch(points, width)


# ---------- shapes ----------
def shape_outline(kind, start, end):
    """Return [(points, closed, always_filled)] for a shape dragged from start to end."""
    a, b = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    (x0, y0), (x1, y1) = np.minimum(a, b), np.maximum(a, b)
    cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
    if kind == "Line":
        return [([a, b], False, False)]
    if kind == "Arrow":
        length = np.linalg.norm(b - a)
        if length < 1:
            return [([a, b], False, False)]
        u = (b - a) / length
        n, head = np.array([-u[1], u[0]]), min(length * 0.4, 14 + length * 0.12)
        return [([a, b - u * head * 0.6], False, False),
                ([b, b - u * head + n * head * 0.55, b - u * head - n * head * 0.55], True, True)]
    if kind == "Rectangle":
        points = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    elif kind == "Circle":
        points = ellipse_points((cx, cy), (rx, ry), count=90)[:-1]
    elif kind == "Triangle":
        points = [(cx, y0), (x1, y1), (x0, y1)]
    elif kind == "Diamond":
        points = [(cx, y0), (x1, cy), (cx, y1), (x0, cy)]
    elif kind == "Star":
        angles = -math.pi / 2 + np.arange(10) * math.pi / 5
        radius = np.where(np.arange(10) % 2, 0.42, 1.0)
        points = np.stack([cx + rx * radius * np.cos(angles), cy + ry * radius * np.sin(angles)], axis=1)
    else:  # Heart
        t = np.linspace(0, 2 * math.pi, 120)[:-1]
        x = 16 * np.sin(t) ** 3
        y = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
        x = x0 + (x - x.min()) / np.ptp(x) * (x1 - x0)
        y = y0 + (y - y.min()) / np.ptp(y) * (y1 - y0)
        points = np.stack([x, y], axis=1)
    return [(points, True, False)]


# ---------- flower stamps ----------
def _direction(angle):
    return np.array([math.cos(angle), math.sin(angle)])


def _stem(p, c, r, rng):
    side = rng.choice((-1, 1))
    bottom = c + (rng.uniform(-0.3, 0.3) * r, 2.6 * r)
    control = c + (-0.45 * r * side, 1.3 * r)
    t = np.linspace(0, 1, 30)[:, None]
    curve = (1 - t) ** 2 * c + 2 * (1 - t) * t * control + t ** 2 * bottom
    p.line(curve, STEM, max(2, r * 0.12))
    for at, flip, size in ((17, 1, 1.0), (24, -1, 0.75)):
        base = curve[at]
        leaf = base + (side * flip * 0.34 * r * size, -0.05 * r)
        p.ellipse(leaf, (0.42 * r * size, 0.15 * r * size), -side * flip * 30, GREEN)
        p.line([base, leaf + (side * flip * 0.3 * r * size, -0.15 * r * size)], STEM, max(1, r * 0.03))


def _daisy(p, c, r, color, spin, rng):
    petal = color or (255, 255, 255)
    edge = shade(color, 0.75) if color else (196, 205, 222)
    for i in range(14):
        a = spin + 2 * math.pi * i / 14
        at = c + 0.55 * r * _direction(a)
        p.ellipse(at, (0.47 * r, 0.15 * r), math.degrees(a), edge)
        p.ellipse(at, (0.42 * r, 0.11 * r), math.degrees(a), petal)
    p.circle(c, 0.27 * r, (234, 138, 12))
    p.circle(c, 0.22 * r, (250, 204, 21))
    p.circle(c - 0.07 * r, 0.07 * r, (254, 240, 138))


def _sunflower(p, c, r, color, spin, rng):
    outer, inner = (shade(color, 0.8), color) if color else ((245, 158, 11), (250, 204, 21))
    for ring, (distance, axes, tint) in enumerate(((0.62, (0.4, 0.13), outer), (0.55, (0.36, 0.12), inner))):
        for i in range(18):
            a = spin + 2 * math.pi * (i + ring / 2) / 18
            p.ellipse(c + distance * r * _direction(a), (axes[0] * r, axes[1] * r), math.degrees(a), tint)
    p.circle(c, 0.4 * r, (120, 53, 15))
    p.circle(c, 0.34 * r, (146, 64, 14))
    k = np.arange(1, 46)
    seeds = c + (0.31 * r * np.sqrt(k / 45))[:, None] * np.stack([np.cos(k * 2.39996 + spin),
                                                                  np.sin(k * 2.39996 + spin)], axis=1)
    p.dots(seeds, 0.028 * r + 0.6, (69, 26, 3))


def _tulip(p, c, r, color, spin, rng):
    tints = ((239, 68, 68), (236, 72, 153), (249, 115, 22), (168, 85, 247))
    body = color or tints[rng.integers(len(tints))]
    dark, light = shade(body, 0.75), shade(body, 1.35)
    p.ellipse(c + (-0.24 * r, 0.02 * r), (0.62 * r, 0.28 * r), 72, dark)
    p.ellipse(c + (0.24 * r, 0.02 * r), (0.62 * r, 0.28 * r), 108, dark)
    p.ellipse(c, (0.66 * r, 0.3 * r), 90, body)
    p.ellipse(c + (0, 0.28 * r), (0.42 * r, 0.32 * r), 0, body)
    p.ellipse(c + (-0.1 * r, 0), (0.36 * r, 0.06 * r), 90, light)


def _rose(p, c, r, color, spin, rng):
    body = color or (225, 29, 72)
    for i in range(5):
        p.circle(c + 0.42 * r * _direction(spin + 2 * math.pi * i / 5), 0.42 * r, shade(body, 0.85))
    p.circle(c, 0.58 * r, body)
    for size, start in ((0.46, 0), (0.33, 120), (0.21, 240), (0.1, 40)):
        arc = ellipse_points(c, (size * r, size * r * 0.9), math.degrees(spin), start, start + 250)
        p.line(arc, shade(body, 0.65), max(1.5, 0.05 * r))


def _blossom(p, c, r, color, spin, rng):
    body = color or (251, 182, 206)
    edge = shade(body, 0.85)
    for i in range(5):
        u = _direction(spin + 2 * math.pi * i / 5)
        p.circle(c + 0.48 * r * u, 0.44 * r, edge)
        p.circle(c + 0.48 * r * u, 0.38 * r, body)
        n = np.array([-u[1], u[0]])
        tip = c + 0.92 * r * u
        p.polygon([tip + n * 0.09 * r, tip - n * 0.09 * r, c + 0.76 * r * u], edge)
    p.circle(c, 0.2 * r, shade(body, 0.7) if color else (219, 39, 119))
    for i in range(10):
        end = c + 0.36 * r * _direction(spin + 2 * math.pi * (i + 0.5) / 10)
        p.line([c, end], (190, 24, 93), max(1, 0.03 * r))
        p.dots([end], 0.05 * r + 0.8, (250, 204, 21))


def _lavender(p, c, r, color, spin, rng):
    body = color or (167, 139, 250)
    dark = shade(body, 0.72)
    top, bottom = c + (0, -r), c + (0, 2.4 * r)
    p.line([top + (0, 0.2 * r), bottom], STEM, max(2, 0.09 * r))
    for side in (-1, 1):
        p.ellipse(bottom + (side * 0.3 * r, -0.7 * r), (0.45 * r, 0.09 * r), 90 + side * 60, GREEN)
    for i in range(10):
        y = top[1] + i * 0.2 * r
        spread = 0.08 * r + i * 0.012 * r
        for side in (-1, 1):
            p.ellipse((c[0] + side * spread, y), (0.16 * r, 0.09 * r), 90 + side * 30,
                      body if (i + (side > 0)) % 2 else dark)
    p.ellipse(top, (0.14 * r, 0.08 * r), 90, body)


def _butterfly(p, c, r, color, spin, rng):
    tints = ((96, 165, 250), (244, 114, 182), (251, 146, 60), (167, 139, 250))
    body = color or tints[rng.integers(len(tints))]
    dark, light = shade(body, 0.6), shade(body, 1.45)
    for side in (-1, 1):
        upper, lower = c + (side * 0.45 * r, -0.28 * r), c + (side * 0.33 * r, 0.3 * r)
        p.ellipse(upper, (0.52 * r, 0.38 * r), -side * 35, dark)
        p.ellipse(upper, (0.46 * r, 0.32 * r), -side * 35, body)
        p.circle(upper + (side * 0.12 * r, -0.05 * r), 0.11 * r, light)
        p.ellipse(lower, (0.34 * r, 0.26 * r), side * 35, dark)
        p.ellipse(lower, (0.29 * r, 0.21 * r), side * 35, body)
        tip = c + (side * 0.25 * r, -0.85 * r)
        p.line([c + (0, -0.55 * r), tip], BODY, max(1, 0.03 * r))
        p.dots([tip], 0.04 * r + 1, BODY)
    p.ellipse(c, (0.5 * r, 0.07 * r), 90, BODY)
    p.circle(c + (0, -0.5 * r), 0.09 * r, BODY)


def _leaf(p, c, r, color, spin, rng):
    body = color or (34, 197, 94)
    u = _direction(spin)
    n = np.array([-u[1], u[0]])
    p.ellipse(c, (0.9 * r, 0.36 * r), math.degrees(spin), shade(body, 0.78))
    p.ellipse(c, (0.84 * r, 0.3 * r), math.degrees(spin), body)
    vein = shade(body, 0.65)
    p.line([c - u * 0.85 * r, c + u * 0.8 * r], vein, max(1, 0.04 * r))
    for k in range(-2, 3):
        base = c + u * k * 0.26 * r
        for side in (-1, 1):
            p.line([base, base + u * 0.16 * r + n * side * 0.22 * r], vein, max(1, 0.025 * r))


_STAMPS = {"Daisy": _daisy, "Sunflower": _sunflower, "Tulip": _tulip, "Rose": _rose,
           "Blossom": _blossom, "Lavender": _lavender, "Butterfly": _butterfly, "Leaf": _leaf}


def flower(painter, kind, center, radius, color=None, stem=True, rng=None):
    """Paint a flower stamp. color=None uses the flower's natural colours."""
    rng = rng or np.random.default_rng()
    c, r = np.asarray(center, dtype=float), radius * rng.uniform(0.9, 1.1)
    if kind == "Tulip" or (stem and kind in ("Daisy", "Sunflower", "Rose", "Blossom")):
        _stem(painter, c, r, rng)
    _STAMPS[kind](painter, c, r, color, rng.uniform(0, 2 * math.pi), rng)


# ---------- boards and banners ----------
def _gradient(stops, vertical=False):
    stops = np.array([rgb(s) for s in stops], dtype=float)
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    t = yy / (HEIGHT - 1) if vertical else xx / (WIDTH - 1) * 0.7 + yy / (HEIGHT - 1) * 0.3
    position = t * (len(stops) - 1)
    index = np.clip(position.astype(int), 0, len(stops) - 2)
    f = (position - index)[..., None]
    return (stops[index] * (1 - f) + stops[index + 1] * f).astype(np.uint8)


def board(name):
    """A 960 × 540 RGB background for the drawing board."""
    image = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
    rng = np.random.default_rng(3)
    if name == "Blackboard":
        smudge = cv2.GaussianBlur(rng.normal(0, 1, (HEIGHT, WIDTH)).astype(np.float32), (0, 0), 25)
        smudge = smudge / (np.abs(smudge).max() + 1e-6) * 8 + rng.normal(0, 2.5, (HEIGHT, WIDTH))
        image = np.clip(np.array((35, 56, 46), dtype=float) + smudge[..., None], 0, 255).astype(np.uint8)
    elif name == "Night sky":
        image = _gradient(("#0f172a", "#1e1b4b", "#312e81"), vertical=True)
        for x, y, s in zip(rng.integers(0, WIDTH, 140), rng.integers(0, HEIGHT, 140), rng.random(140)):
            cv2.circle(image, (int(x), int(y)), 1 if s < 0.85 else 2, (230, 230, 255), -1, cv2.LINE_AA)
    elif name == "Grid paper":
        image[:] = (252, 252, 255)
        for step, color in ((24, (225, 232, 246)), (120, (196, 210, 238))):
            image[::step, :] = color
            image[:, ::step] = color
    elif name == "Dot grid":
        image[:] = (255, 252, 245)
        for y in range(12, HEIGHT, 24):
            for x in range(12, WIDTH, 24):
                cv2.circle(image, (x, y), 2, (200, 200, 214), -1, cv2.LINE_AA)
    elif name == "Lined paper":
        image[:] = (255, 255, 250)
        image[64::32, :] = (186, 215, 240)
        image[:, 78:80] = (244, 163, 163)
    elif name == "Pastel":
        image = _gradient(("#ede9fe", "#fce7f3", "#dbeafe"))
    elif name == "Sunset":
        image = _gradient(("#fed7aa", "#fbcfe8", "#ddd6fe"), vertical=True)
    elif name == "Ocean":
        image = _gradient(("#cffafe", "#bfdbfe", "#c7d2fe"))
    elif name == "Mint":
        image = _gradient(("#dcfce7", "#ccfbf1", "#e0f2fe"))
    return image


def banner(width, height, stops, background, radius=18):
    """A rounded, gradient header strip with soft decorative bubbles."""
    colors = np.array([rgb(s) for s in stops], dtype=float)
    t = np.linspace(0, 1, width) * (len(colors) - 1)
    index = np.clip(t.astype(int), 0, len(colors) - 2)
    f = (t - index)[:, None]
    row = colors[index] * (1 - f) + colors[index + 1] * f
    strip = Image.fromarray(np.repeat(row[None], height, axis=0).astype(np.uint8)).convert("RGBA")
    bubbles = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(bubbles)
    rng = np.random.default_rng(11)
    for _ in range(14):
        x, y, s = rng.integers(0, width), rng.integers(-20, height + 20), rng.integers(14, 60)
        draw.ellipse((x - s, y - s, x + s, y + s), fill=(255, 255, 255, int(rng.integers(18, 46))))
    strip = Image.alpha_composite(strip, bubbles).convert("RGB")
    shape = Image.new("L", (width, height), 0)
    ImageDraw.Draw(shape).rounded_rectangle((0, 0, width - 1, height - 1), radius, fill=255)
    return Image.composite(strip, Image.new("RGB", (width, height), background), shape)
