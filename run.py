"""Create the environment, install dependencies when needed, and launch AirInk."""

import os
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent


def main():
    if sys.version_info[:2] != (3, 12):
        raise SystemExit("Use Python 3.12: python3.12 run.py (Mac) or py -3.12 run.py (Windows).")
    environment = ROOT / ".venv"
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    requirements = (ROOT / "requirements.txt").read_bytes()
    stamp = environment / ".airink-requirements"
    if not python.exists():
        print("Creating AirInk's Python environment...", flush=True)
        venv.EnvBuilder(with_pip=True).create(environment)
    if not stamp.exists() or stamp.read_bytes() != requirements:
        print("Installing AirInk dependencies...", flush=True)
        # Environments made by other tools (e.g. uv) may lack pip.
        subprocess.run([str(python), "-m", "ensurepip", "--upgrade"],
                       check=True, stdout=subprocess.DEVNULL)
        subprocess.run([str(python), "-m", "pip", "install", "-r",
                        str(ROOT / "requirements.txt")], check=True)
        stamp.write_bytes(requirements)
    subprocess.run([str(python), str(ROOT / "main.py")], cwd=ROOT, check=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode)
