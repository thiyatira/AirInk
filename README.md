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

- **Start camera:** allow camera access; keep your hand upright and fully in frame.
- **Only index finger raised:** write. Keep the thumb and other fingers folded.
- **Other poses:** pause.
- **Pen / Eraser, colour, brush size:** change your drawing tool.
- **Opacity / Board:** fade the camera or switch to a white board.
- **Save / Open:** save a PNG or reopen it to continue. Each save creates a new copy.
- **Clear:** erase the canvas. **Esc:** exit. Camera off: draw with the mouse.

Save before opening another drawing or clearing the canvas.

## Saved drawings

- **Mac:** `~/Library/Application Support/AirInk/drawings/`
- **Windows:** `%LOCALAPPDATA%\AirInk\drawings\`

**Open** starts in this folder. The report and presentation (PDF + editable files) are in `documents/`.
