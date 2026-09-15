"""Launch labelImg on the test set, with the environment it needs.

Two things break labelImg in a conda install and both are silent:

1. Qt finds its image-format plugins but cannot load qjpeg.dll, because that plugin's own
   dependency (libjpeg) lives in <prefix>/Library/bin, which is not on PATH unless the
   environment is activated. Qt reports no error - JPEG simply disappears from the supported
   formats, so `Open Dir` scans the folder, matches nothing, and shows an empty file list.
2. labelImg 1.8.6 passes float coordinates to QPainter.drawRect/drawLine, which modern PyQt5
   rejects, so the canvas raises as soon as you start drawing a box.

The JPEG check runs in a child process on purpose: it needs a QApplication to load the plugins,
Qt permits only one per process, and labelImg creates its own. Checking in-process would leave
labelImg with a dead application object and it would exit with "Please instantiate the
QApplication object first".

Run:  python scripts/annotate.py
"""
from __future__ import annotations

import inspect
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTSET = os.path.join(ROOT, "testset")

PROBE = (
    "import sys;"
    "from PyQt5.QtWidgets import QApplication;"
    "from PyQt5.QtGui import QImageReader;"
    "app = QApplication(sys.argv[:1]);"
    "fmts = sorted(f.data().decode().lower() for f in QImageReader.supportedImageFormats());"
    "print(','.join(fmts))"
)


def fix_dll_path():
    """Put the conda library directories where Qt's plugin loader will look."""
    added = []
    for rel in (("Library", "bin"), ("Library", "mingw-w64", "bin"), ("Library", "lib")):
        path = os.path.join(sys.prefix, *rel)
        if os.path.isdir(path) and path not in os.environ["PATH"]:
            os.environ["PATH"] = path + os.pathsep + os.environ["PATH"]
            added.append(path)
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(path)
                except OSError:
                    pass
    return added


def supported_formats():
    """Ask a child process which image formats Qt can actually read."""
    try:
        out = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True,
                             timeout=120, env=os.environ)
    except subprocess.TimeoutExpired:
        return None
    if out.returncode != 0:
        return None
    return set((out.stdout.strip().splitlines() or [""])[-1].split(","))


def check_canvas_patch():
    """labelImg 1.8.6 hands floats to QPainter; report if this copy is still unpatched."""
    try:
        import libs.canvas as canvas
        src = inspect.getsource(canvas)
    except (ImportError, OSError):
        return None
    return [line.strip() for line in src.splitlines()
            if ("drawRect(" in line or "drawLine(" in line) and "int(" not in line]


def main():
    added = fix_dll_path()
    if added:
        print("added to PATH:")
        for path in added:
            print(f"  {path}")

    formats = supported_formats()
    if formats is None:
        sys.exit("could not query Qt image formats - is PyQt5 installed in this interpreter?")
    if "jpg" not in formats:
        sys.exit(
            "Qt cannot read JPEG - the qjpeg plugin failed to load.\n"
            f"  plugin dir: {os.path.join(sys.prefix, 'Library', 'plugins', 'imageformats')}\n"
            "  labelImg would show an empty file list. Launch from an activated conda\n"
            "  environment, or reinstall with: conda install -c conda-forge qt-main"
        )
    print(f"JPEG support: OK ({len(formats)} formats)")

    unpatched = check_canvas_patch()
    if unpatched:
        print("\nWARNING: libs/canvas.py still passes floats to QPainter; drawing a box will")
        print("         raise TypeError. Wrap these coordinates in int():")
        for line in unpatched:
            print(f"           {line}")

    images = os.path.join(TESTSET, "images")
    labels = os.path.join(TESTSET, "labels")
    classes = os.path.join(labels, "classes.txt")
    for path in (images, labels, classes):
        if not os.path.exists(path):
            sys.exit(f"missing {path} - run scripts/prepare_testset.py first")

    count = len([f for f in os.listdir(images) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    print(f"\nopening {count} images from {images}")
    print(f"  saving annotations to {labels}")
    print("\nSet the format button in the left toolbar to YOLO before saving:")
    print("  labelImg loads .txt regardless of format, but SAVES PascalVOC XML by default,")
    print("  which would leave your work out of the .txt files entirely.")

    # Hand over to labelImg in this same process, having created no QApplication of our own.
    sys.argv = [sys.argv[0], images, classes, labels]
    from labelImg.labelImg import main as labelimg_main

    return labelimg_main()


if __name__ == "__main__":
    sys.exit(main())
