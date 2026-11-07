# AirInk

Write in the air with your index finger. One Python file, a webcam, and a small Tkinter interface.

## One-command setup from GitHub

Install **Git** and **Python 3.12 with Tkinter** first. Use the standard Python installer with Tcl/Tk enabled. On Windows, include the Python launcher (`py`).

This repository is public; no GitHub sign-in is needed to download it. These commands clone into a new `AirInk` folder, install the project dependencies and launch the app.

macOS — Terminal:

```bash
git clone https://github.com/thiyatira/AirInk.git AirInk && cd AirInk && python3.12 run.py
```

Windows — PowerShell:

```powershell
git clone https://github.com/thiyatira/AirInk.git AirInk; if ($LASTEXITCODE -eq 0) { Set-Location AirInk; py -3.12 run.py }
```

`run.py` creates a local `.venv` and installs `requirements.txt`. Later launches reuse it; changed requirements trigger installation again. Setup stops if installation fails.

## Run again or use the ZIP

Open a terminal inside the project folder. On macOS run `python3.12 run.py`; on Windows run `py -3.12 run.py`.

To update a GitHub checkout and launch in one command:

```bash
# macOS
git pull --ff-only && python3.12 run.py
```

```powershell
# Windows PowerShell
git pull --ff-only; if ($LASTEXITCODE -eq 0) { py -3.12 run.py }
```

The hand model is included in `assets/`. Internet is needed for GitHub and dependency installation; normal drawing and storage work offline. On Linux, install your distribution's Python Tkinter package if missing.

## Use

1. Click **Start camera** and allow camera access.
2. Face your palm toward the camera, with fingers pointing up.
3. Raise **only the index finger** to write. Keep all other fingers, including the thumb, folded.
4. Raise **two or more fingers** to move without drawing. A fist or any other pose also pauses.
5. Choose a colour, brush size, or **Eraser**. Use the same index-finger pose to erase.
6. Switch between **Camera** and **Board**. Use **Camera opacity** to fade the camera from 100% to 0%; ink stays fully visible. The default is 60%.
7. Adjust **Camera zoom** (1×–2.5×; default 1.5×). Only the display is cropped; tracking uses the full camera frame.
8. **Save** stores a PNG in your local drawings folder. **Open** loads a saved drawing so you can continue writing or erasing.

**Clear** removes the drawing. **Esc** closes the app. With the camera off, drag with the mouse to draw.

## Saved drawings

| Platform | Fixed folder |
| --- | --- |
| macOS | `~/Library/Application Support/AirInk/drawings/` |
| Windows | `%LOCALAPPDATA%\AirInk\drawings\` |
| Linux | `~/.local/share/AirInk/drawings/` |

The folder is created automatically. **Open** starts in this folder. Select a PNG, continue drawing, then **Save** creates a new timestamped copy. Earlier saves stay intact, including after app restarts or project updates. Files stay on your computer.

PNGs have a white background and contain only the drawing. When reopened, white becomes empty canvas, so the camera can show through and the eraser still works. Open accepts AirInk-sized PNGs (960 × 540). Save before opening another drawing or clearing the canvas; unsaved work is replaced.

## Remember the code

`camera → landmarks → pose → fingertip → line → screen`

- `writing_pose()` checks that only the index finger is raised.
- `tick()` reads the camera, tracks the hand, and refreshes the screen.
- `camera_view()` crops the display; `view_point()` maps the fingertip into it.
- `draw()` joins fingertip positions; `compose()` combines ink and background.
- `build_ui()` creates the controls; `save()` and `open_drawing()` store and restore PNGs.

The pose must last three frames before drawing starts. A short smoothing step reduces jitter. Lost tracking resets the stroke so separate letters do not get connected.

## Demo tips

Keep one hand inside the full camera frame, upright, and well lit. Zoom lets parts of your hand sit outside the displayed crop while still being tracked. It cannot recover fingers outside the actual camera frame. If the fingertip leaves the crop, writing pauses; move it back to the centre. Start with a broad “Hi”, lift two fingers between strokes, then switch to Board and save. If the camera is busy or denied, close other camera apps or allow access for the terminal/IDE in system settings. Mouse drawing is available as a fallback.

## Files

```text
main.py                       App
run.py                        Dependency setup and launcher
requirements.txt              Dependencies
assets/                       Hand-tracking model
documents/
  presentation/               Final PDF + editable PowerPoint (.pptx)
  report/                     Final PDF + editable Word (.docx)
  screenshots/                App captures and sample drawing
  fonts/                      Presentation fonts and licences
AirInk_Project.zip            Complete project to share
```

The 11-slide presentation includes speaker notes. The 7-page report is available as PDF and editable Word. For the same slide appearance on another computer, install the fonts in `documents/fonts/`, or present the PDF.

All 38 automated checks passed on macOS, covering gestures, zoom, opacity, storage, reopening and setup failures. Windows storage paths were verified with simulated platform settings. Live webcam and native Windows checks still need to be done on the presentation computer.
