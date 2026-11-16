# AirInk

Write with your index finger using a webcam.

## Install and run

Requires **Git** and **Python 3.12 with Tkinter**. Dependencies install automatically.

**macOS — Terminal**

```bash
git clone https://github.com/thiyatira/AirInk.git AirInk && cd AirInk && python3.12 run.py
```

**Windows — PowerShell**

```powershell
git clone https://github.com/thiyatira/AirInk.git AirInk; if ($LASTEXITCODE -eq 0) { Set-Location AirInk; py -3.12 run.py }
```

To run again, open the project folder in a terminal:

- **Mac:** `python3.12 run.py`
- **Windows:** `py -3.12 run.py`

## Controls

**Hand gestures** (Start camera, keep your hand upright and fully in frame)

- **☝ Only index finger raised:** use the current tool. Keep the thumb and other fingers folded.
- **✋ Open hand / ✊ fist / other poses:** pause.
- **✌ Hold index + middle finger for about a second:** undo.

**Tools** (left panel)

- **Draw:** 9 brushes — Pen, Marker, Highlighter, Neon glow, Rainbow, Dotted, Spray, Calligraphy, Chalk.
- **Eraser:** rub out ink; Size sets its width.
- **Shapes:** drag a Line, Arrow, Rectangle, Circle, Triangle, Star, Heart or Diamond. *Fill* fills it.
- **Flowers:** place Daisy, Sunflower, Tulip, Rose, Blossom, Lavender, Butterfly or Leaf stamps.
  *Natural* uses real flower colours (off = your colour), *Stem* adds stems, *Scatter* plants a trail as you move.
- **Undo / Redo / Clear:** 30 undo steps; Clear can be undone too.

**Style** (right panel and header)

- 12 colours, a hex box and a full colour picker; Size (`[` / `]` keys) and Ink opacity.
- **Theme:** Pastel Dream, Neon Night, Sunny Vibes, Ocean Glass, Classic.
- **Board:** White, Blackboard, Night sky, Grid/Dot/Lined paper, Pastel, Sunset, Ocean, Mint.
- **Camera view / fade:** show the camera behind the ink, or only the board.

**Files**

- **Save PNG:** saves the drawing on its board. **Snapshot:** saves exactly what is on screen, camera included.
- **Gallery / Open:** reopen a saved drawing and keep editing.
- Shortcuts: Cmd/Ctrl+Z undo, Cmd/Ctrl+Shift+Z or Ctrl+Y redo, Cmd/Ctrl+S save, Esc exit. Camera off: draw with the mouse.

## Saved drawings

- **Mac:** `~/Library/Application Support/AirInk/drawings/`
- **Windows:** `%LOCALAPPDATA%\AirInk\drawings\`

Snapshots go to the sibling `snapshots/` folder. **Open** starts in the drawings folder.

Run the checks with `.venv/bin/python -m unittest discover tests` (Windows: `.venv\Scripts\python -m unittest discover tests`). The report and presentation (PDF + editable files) are in `documents/`.
