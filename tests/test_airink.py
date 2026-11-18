"""AirInk checks. Run from the project folder: .venv/bin/python -m unittest discover tests

MySQL checks use a separate database (airink_test) that is created and dropped here,
and are skipped when MySQL is not running.
"""

import os
import shutil
import sys
import tempfile
import time
import tkinter as tk
import unittest
from pathlib import Path
from types import SimpleNamespace as P

import numpy as np
from PIL import Image

os.environ["DB_NAME"] = "airink_test"  # never touch the real airink_db
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import art  # noqa: E402
import db  # noqa: E402
import main  # noqa: E402


def drop_test_database():
    server = {k: v for k, v in db.DB_CONFIG.items() if k != "database"}
    conn = db.mysql.connector.connect(connection_timeout=3, **server)
    try:
        conn.cursor().execute("DROP DATABASE IF EXISTS airink_test")
    finally:
        conn.close()


try:
    drop_test_database()
    MYSQL = True
except db.Error:
    MYSQL = False
needs_mysql = unittest.skipUnless(MYSQL, "MySQL is not running")


def hand(raised, x=0.5, y=0.5, thumb_open=False):
    """Fake MediaPipe landmarks: raised = [index, middle, ring, little]."""
    points = [P(x=x, y=y + 0.3) for _ in range(21)]
    points[17] = P(x=x - 0.1, y=y + 0.25)
    points[3] = P(x=x + 0.05, y=y + 0.2)
    points[4] = P(x=x + 0.3, y=y + 0.2) if thumb_open else P(x=x + 0.06, y=y + 0.22)
    for tip, up in zip((8, 12, 16, 20), raised):
        points[tip - 2] = P(x=x, y=y + 0.1)
        points[tip] = P(x=x, y=y if up else y + 0.15)
    return points


class FakeCamera:
    def read(self):
        return True, np.zeros((540, 960, 3), np.uint8)

    def release(self):
        pass


class FakeTracker:
    pose = None

    def detect_for_video(self, image, timestamp):
        return P(hand_landmarks=[self.pose] if self.pose else [])

    def close(self):
        pass


class AirInkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        base = Path(cls.folder.name)
        main.app_folder = lambda: base
        main.storage_folder = lambda: base / "drawings"
        cls.root = tk.Tk()
        cls.root.withdraw()

    @classmethod
    def tearDownClass(cls):
        cls.root.destroy()
        cls.folder.cleanup()
        if MYSQL:
            drop_test_database()

    def setUp(self):
        for child in self.root.winfo_children():
            child.destroy()
        shutil.rmtree(main.storage_folder(), ignore_errors=True)
        if MYSQL:
            db.init_db()
            db.run("DELETE FROM drawings")
            db.run("DELETE FROM sessions")
        self.app = main.AirInk(self.root)
        self.root.after_cancel(self.app.after_id)

    def stroke(self, points):
        self.app.begin(points[0])
        for point in points[1:]:
            self.app.move(point)
        self.app.end()

    def line(self, y=200):
        return [(100 + i * 10, y) for i in range(30)]

    # ---------- gestures ----------
    def test_gesture_names(self):
        self.assertEqual(main.gesture(hand([1, 0, 0, 0])), "point")
        self.assertEqual(main.gesture(hand([1, 0, 0, 0], thumb_open=True)), "other")
        self.assertEqual(main.gesture(hand([1, 1, 0, 0])), "peace")
        self.assertEqual(main.gesture(hand([1, 1, 1, 1])), "palm")
        self.assertEqual(main.gesture(hand([0, 0, 0, 0])), "fist")

    def camera(self):
        self.app.camera, self.app.tracker = FakeCamera(), FakeTracker()
        return self.app.tracker

    def test_three_frames_start_drawing_and_palm_pauses(self):
        tracker = self.camera()
        tracker.pose = hand([1, 0, 0, 0])
        self.app.camera_step()
        self.app.camera_step()
        self.assertIsNone(self.app.active)
        self.app.camera_step()
        self.assertEqual(self.app.active, "Draw")
        tracker.pose = hand([1, 1, 1, 1])
        for _ in range(3):
            self.app.camera_step()
        self.assertIsNone(self.app.active)
        self.assertTrue(self.app.mask.any())

    def test_holding_two_fingers_undoes_once(self):
        self.stroke(self.line())
        self.stroke(self.line(300))
        tracker = self.camera()
        tracker.pose = hand([1, 1, 0, 0])
        start = time.monotonic()
        while time.monotonic() - start < main.UNDO_HOLD * 2:
            self.app.camera_step()
            time.sleep(0.02)
        self.assertEqual(len(self.app.undo_stack), 1)  # only one undo while held

    # ---------- brushes ----------
    def test_every_brush_leaves_ink(self):
        for brush in main.BRUSHES:
            with self.subTest(brush=brush):
                self.app.clear()
                self.app.brush.set(brush)
                self.stroke(self.line())
                self.assertTrue(self.app.mask.any())

    def test_highlighter_is_even_where_it_crosses_itself(self):
        self.app.brush.set("Highlighter")
        self.stroke([(100, 200), (400, 200), (400, 100), (250, 100), (250, 300)])
        self.assertLess(self.app.mask.max(), 128)
        self.assertEqual(self.app.mask[200, 250], self.app.mask[200, 150])

    def test_neon_glow_spreads_beyond_the_core(self):
        self.app.brush.set("Neon")
        self.stroke(self.line())
        self.assertGreater(self.app.mask[212, 200], 0)  # 12 px from a 6 px line
        self.app.clear()
        self.app.brush.set("Pen")
        self.stroke(self.line())
        self.assertEqual(self.app.mask[212, 200], 0)

    def test_rainbow_changes_colour_along_the_stroke(self):
        self.app.brush.set("Rainbow")
        self.stroke([(100 + i * 20, 200) for i in range(40)])
        self.assertFalse(np.array_equal(self.app.ink[200, 110], self.app.ink[200, 800]))

    def test_dotted_leaves_gaps(self):
        self.app.brush.set("Dotted")
        self.stroke(self.line())
        row = self.app.mask[200, 100:390] > 0
        self.assertTrue(row.any() and not row.all())

    def test_opacity_slider_makes_ink_translucent(self):
        self.app.alpha.set(50)
        self.stroke(self.line())
        self.assertAlmostEqual(int(self.app.mask[200, 200]), 128, delta=2)

    def test_eraser_removes_ink(self):
        self.stroke(self.line())
        self.app.tool.set("Eraser")
        self.stroke(self.line())
        self.assertFalse(self.app.mask[200, 120:370].any())

    # ---------- shapes and flowers ----------
    def test_every_shape_draws(self):
        self.app.tool.set("Shapes")
        for shape in art.SHAPES:
            with self.subTest(shape=shape):
                self.app.clear()
                self.app.shape.set(shape)
                self.stroke([(300, 150), (500, 300)])
                self.assertTrue(self.app.mask.any())

    def test_fill_fills_the_inside(self):
        self.app.tool.set("Shapes")
        self.app.shape.set("Rectangle")
        self.stroke([(300, 150), (500, 300)])
        self.assertEqual(self.app.mask[225, 400], 0)
        self.app.fill.set(True)
        self.stroke([(300, 150), (500, 300)])
        self.assertEqual(self.app.mask[225, 400], 255)

    def test_click_without_drag_makes_no_shape(self):
        self.app.tool.set("Shapes")
        self.stroke([(300, 150)])
        self.assertFalse(self.app.mask.any())
        self.assertEqual(self.app.undo_stack, [])

    def test_every_flower_stamps(self):
        self.app.tool.set("Flowers")
        for kind in art.FLOWERS:
            with self.subTest(flower=kind):
                self.app.clear()
                self.app.flower.set(kind)
                self.stroke([(480, 200)])
                self.assertTrue(self.app.mask.any())

    def test_natural_colours_can_be_turned_off(self):
        self.app.tool.set("Flowers")
        self.app.flower.set("Leaf")
        self.app.natural.set(False)
        self.app.choose_color("#ec4899")
        self.stroke([(480, 270)])
        colours, counts = np.unique(self.app.ink[self.app.mask == 255], axis=0, return_counts=True)
        self.assertEqual(tuple(colours[counts.argmax()]), art.rgb("#ec4899"))

    def test_scatter_plants_a_trail(self):
        self.app.tool.set("Flowers")
        self.app.flower.set("Leaf")
        self.app.trail.set(True)
        self.stroke([(100 + i * 10, 270) for i in range(70)])
        filled = np.flatnonzero(self.app.mask[:, 100:800].any(axis=0))
        self.assertGreater(len(filled), 300)  # one leaf is under 100 px wide
        self.assertEqual(len(self.app.undo_stack), 1)  # one undo removes the whole trail

    # ---------- undo ----------
    def test_undo_and_redo(self):
        self.stroke(self.line())
        drawn = self.app.mask.copy()
        self.app.undo()
        self.assertFalse(self.app.mask.any())
        self.app.redo()
        self.assertTrue(np.array_equal(self.app.mask, drawn))

    def test_new_stroke_clears_redo(self):
        self.stroke(self.line())
        self.app.undo()
        self.stroke(self.line(300))
        self.assertEqual(self.app.redo_stack, [])

    def test_undo_keeps_the_last_30_steps(self):
        for i in range(40):
            self.stroke([(10 + i * 20, 100), (10 + i * 20, 120)])
        self.assertEqual(len(self.app.undo_stack), main.UNDO_LIMIT)

    def test_clear_can_be_undone(self):
        self.stroke(self.line())
        self.app.clear()
        self.assertFalse(self.app.mask.any())
        self.app.undo()
        self.assertTrue(self.app.mask.any())

    # ---------- files ----------
    def test_save_and_reopen_on_a_dark_board(self):
        self.app.board.set("Blackboard")
        self.app.alpha.set(60)
        self.stroke(self.line())
        ink, mask = self.app.ink.copy(), self.app.mask.copy()
        self.app.save()
        saved = sorted(main.storage_folder().glob("AirInk-*.png"))[-1]
        self.app.clear()
        self.app.load_drawing(saved)
        self.assertTrue(np.array_equal(self.app.mask, mask))
        self.assertTrue(np.array_equal(self.app.ink[mask > 0], ink[mask > 0]))

    def test_older_white_png_still_opens(self):
        picture = np.full((540, 960, 3), 255, np.uint8)
        picture[100:120, 100:300] = (37, 99, 235)
        path = Path(self.folder.name) / "old.png"
        Image.fromarray(picture).save(path)
        self.app.load_drawing(path)
        self.assertEqual(self.app.mask[110, 200], 255)
        self.assertEqual(self.app.mask[300, 500], 0)

    def test_snapshot_saves_a_picture(self):
        self.app.snapshot()
        self.assertEqual(len(list((Path(self.folder.name) / "snapshots").glob("*.png"))), 1)

    # ---------- interface ----------
    def test_every_board_renders(self):
        for name in art.BOARDS:
            with self.subTest(board=name):
                self.assertEqual(art.board(name).shape, (540, 960, 3))

    def test_every_theme_and_tool_fits_the_panels(self):
        for theme in main.THEMES:
            self.app.theme.set(theme)
            self.app.build_ui()
            for tool in ("Draw", "Eraser", "Shapes", "Flowers"):
                with self.subTest(theme=theme, tool=tool):
                    self.app.tool.set(tool)
                    self.app.show_options()
                    self.root.update_idletasks()
                    self.assertLessEqual(self.root.winfo_reqwidth(), main.TOTAL)
                    for panel in (self.app.screen.master.winfo_children()[0], self.app.options.master):
                        bottom = max(c.winfo_y() + c.winfo_height() for c in panel.winfo_children())
                        self.assertLessEqual(bottom, panel.winfo_height())
                        for child in panel.winfo_children():
                            self.assertLessEqual(child.winfo_reqwidth(), panel.winfo_width())

    def test_typed_hex_colour(self):
        self.app.hex.set("#f6a")
        self.app.typed_color()
        self.assertEqual(self.app.color, "#ff66aa")
        self.app.hex.set("nope")
        self.app.typed_color()
        self.assertEqual(self.app.color, "#ff66aa")


    # ---------- MySQL ----------
    @needs_mysql
    def test_save_adds_a_drawing_row(self):
        self.app.theme.set("Neon Night")
        self.app.board.set("Blackboard")
        self.stroke(self.line())
        self.app.save()
        rows, total = self.app.find_drawings()
        self.assertEqual(total, 1)
        _, title, theme, board, _, file_name = rows[0]
        self.assertTrue(title.startswith("Drawing "))
        self.assertEqual((theme, board), ("Neon Night", "Blackboard"))
        self.assertTrue((main.storage_folder() / file_name).exists())

    @needs_mysql
    def test_search_by_title_board_and_date(self):
        self.stroke(self.line())
        self.app.save()
        self.app.board.set("Grid paper")
        self.app.save()
        rows, _ = self.app.find_drawings()
        self.app.rename_drawing(rows[0][5], "Sunflower garden")
        self.assertEqual(len(self.app.find_drawings("garden")[0]), 1)
        self.assertEqual(len(self.app.find_drawings("Grid paper")[0]), 1)
        self.assertEqual(len(self.app.find_drawings(f"{main.datetime.now():%Y-%m-%d}")[0]), 2)
        self.assertEqual(len(self.app.find_drawings("no such drawing")[0]), 0)

    @needs_mysql
    def test_delete_removes_the_row_and_files(self):
        self.stroke(self.line())
        self.app.save()
        file_name = self.app.find_drawings()[0][0][5]
        self.app.delete_drawing(file_name)
        self.assertEqual(self.app.find_drawings()[1], 0)
        self.assertFalse((main.storage_folder() / file_name).exists())
        self.assertFalse((main.storage_folder() / ".layers" / file_name).exists())

    @needs_mysql
    def test_older_files_join_the_gallery(self):
        main.storage_folder().mkdir(parents=True)
        Image.new("RGB", (960, 540), "white").save(main.storage_folder() / "AirInk-20250101-120000-000000.png")
        rows, total = self.app.find_drawings()
        self.assertEqual(total, 1)
        self.assertTrue(rows[0][1].startswith("Drawing "))

    @needs_mysql
    def test_session_counts_are_recorded_in_one_row(self):
        self.stroke(self.line())
        self.stroke(self.line(300))
        self.app.tool.set("Shapes")
        self.stroke([(300, 150), (500, 300)])
        self.app.tool.set("Flowers")
        self.stroke([(480, 200)])
        self.app.tool.set("Eraser")
        self.stroke(self.line())
        self.app.undo()
        self.app.save()  # saving also records the session
        self.app.record_session()
        rows = db.run("SELECT strokes, shapes, flowers, erases, undos, saves, used_camera FROM sessions", fetch=True)
        self.assertEqual(rows, [(2, 1, 1, 1, 1, 1, 0)])

    @needs_mysql
    def test_no_session_row_without_activity(self):
        self.app.record_session()
        self.assertEqual(db.run("SELECT COUNT(*) FROM sessions", fetch=True)[0][0], 0)

    @needs_mysql
    def test_gallery_and_stats_windows_open(self):
        self.stroke(self.line())
        self.app.save()
        self.app.gallery()
        self.app.show_stats()
        windows = [w for w in self.root.winfo_children() if isinstance(w, tk.Toplevel)]
        self.assertEqual([w.title() for w in windows], ["AirInk Gallery", "AirInk — My stats"])

    def test_app_still_works_without_mysql(self):
        saved, shown = dict(db.DB_CONFIG), []
        db.DB_CONFIG.update(host="127.0.0.1", port=1)  # nothing listens here
        original = main.messagebox.showinfo
        main.messagebox.showinfo = lambda *args, **kwargs: shown.append(args)
        try:
            app = main.AirInk(self.root)
            self.root.after_cancel(app.after_id)
            self.assertIsNotNone(app.db_error)
            app.begin((100, 100))
            app.move((300, 300))
            app.end()
            app.save()
            self.assertEqual(len(list(main.storage_folder().glob("AirInk-*.png"))), 1)
            app.gallery()
            app.show_stats()
            self.assertEqual(len(shown), 1)  # "Stats are stored in MySQL…"
        finally:
            db.DB_CONFIG.clear()
            db.DB_CONFIG.update(saved)
            main.messagebox.showinfo = original


if __name__ == "__main__":
    unittest.main()
