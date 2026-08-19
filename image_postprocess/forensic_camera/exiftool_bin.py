"""Locate ExifTool. MakerNotes cannot be created without the binary."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

_PACKAGE = Path(__file__).resolve().parent
_REPO = _PACKAGE.parents[1]


def find_exiftool() -> str | None:
    env = os.environ.get("EXIFTOOL_PATH")
    if env and Path(env).is_file():
        return env
    for name in ("exiftool", "exiftool.exe", "ExifTool.exe"):
        hit = shutil.which(name)
        if hit:
            return hit
    candidates = [
        _REPO / "image_postprocess" / "bin" / "ExifTool.exe",
        _REPO / "image_postprocess" / "bin" / "exiftool",
        _REPO / "tools" / "exiftool" / "ExifTool.exe",
        Path(__file__).resolve().parents[3] / "tools" / "exiftool" / "ExifTool.exe",
    ]
    for path in candidates:
        if path.is_file():
            return str(path)
    return None


def run_exiftool(args: list[str], *, exe: str | None = None) -> None:
    binary = exe or find_exiftool()
    if not binary:
        raise FileNotFoundError(
            "ExifTool not found. Install it (PATH) or set EXIFTOOL_PATH. "
            "MakerNote copy requires the binary; piexif cannot write Apple MakerNotes."
        )
    proc = subprocess.run(
        [binary, "-overwrite_original", "-q", "-q", *args],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout or "exiftool failed")
