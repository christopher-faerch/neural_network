"""Gesture artwork: label -> image file mapping with a drawn placeholder when no artwork exists.

Drop files named after the model labels into app/assets/gestures/ (e.g. `extension.svg` or
`extension.png`, see GESTURE_IMAGES) and they are used automatically; nothing else changes.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPixmap

from app import theme
from network import config as C

ASSETS_DIR = Path(__file__).resolve().parent / "assets" / "gestures"

# model label -> preferred image file name (first existing of .svg / .png is used)
GESTURE_IMAGES: dict[str, str] = {label: label for label in C.GESTURES}

PRETTY_NAMES: dict[str, str] = {
    "neutral": "Rest / neutral", "extension": "Wrist extension", "flexion": "Wrist flexion",
    "ulnar_dev": "Ulnar deviation", "radial_dev": "Radial deviation", "grip": "Grip",
    "abduction": "Finger abduction", "adduction": "Finger adduction", "supination": "Supination", "pronation": "Pronation",
}
ABBREVIATIONS: dict[str, str] = {
    "neutral": "REST", "extension": "EXT", "flexion": "FLX", "ulnar_dev": "ULN", "radial_dev": "RAD",
    "grip": "GRIP", "abduction": "ABD", "adduction": "ADD", "supination": "SUP", "pronation": "PRO",
}


def pretty_name(label: str) -> str:
    return PRETTY_NAMES.get(label, label.replace("_", " "))


def image_path(label: str) -> Path | None:
    stem = GESTURE_IMAGES.get(label, label)
    for suffix in (".svg", ".png"):
        p = ASSETS_DIR / f"{stem}{suffix}"
        if p.exists():
            return p
    return None


def gesture_pixmap(label: str, size: QSize) -> QPixmap:
    """Artwork for `label` scaled into `size`, or a drawn placeholder (hand silhouette + abbreviation)."""
    path = image_path(label)
    if path is not None:
        pix = _load_scaled(path, size)
        if pix is not None:
            return pix
    return _placeholder(label, size)


def _load_scaled(path: Path, size: QSize) -> QPixmap | None:
    if path.suffix == ".svg":
        try:
            from PySide6.QtSvg import QSvgRenderer
        except ImportError:
            return None
        renderer = QSvgRenderer(str(path))
        if not renderer.isValid():
            return None
        pix = QPixmap(size)
        pix.fill(Qt.transparent)
        painter = QPainter(pix)
        renderer.render(painter, QRectF(0, 0, size.width(), size.height()))
        painter.end()
        return pix
    pix = QPixmap(str(path))
    if pix.isNull():
        return None
    return pix.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation)


def _placeholder(label: str, size: QSize) -> QPixmap:
    """A simple hand silhouette drawn with QPainter, tinted in the accent colour, plus the abbreviation."""
    w, h = size.width(), size.height()
    pix = QPixmap(size)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    accent = QColor(theme.ACCENT)
    fill = QColor(accent)
    fill.setAlpha(60)
    # palm + four fingers + thumb, proportional to the pixmap
    path = QPainterPath()
    s = min(w, h)
    cx, base = w / 2, h * 0.70                         # hand in the upper 70 %, abbreviation below it
    palm_w, palm_h = s * 0.40, s * 0.32
    path.addRoundedRect(QRectF(cx - palm_w / 2, base - palm_h, palm_w, palm_h), s * 0.08, s * 0.08)
    finger_w = palm_w / 4.6
    heights = (0.22, 0.30, 0.32, 0.26)
    for i, fh in enumerate(heights):
        x = cx - palm_w / 2 + i * (palm_w / 4) + (palm_w / 4 - finger_w) / 2
        path.addRoundedRect(QRectF(x, base - palm_h - s * fh + s * 0.03, finger_w, s * fh), finger_w / 2, finger_w / 2)
    path.addRoundedRect(QRectF(cx - palm_w / 2 - finger_w * 0.95, base - palm_h * 0.75, finger_w, s * 0.24), finger_w / 2, finger_w / 2)
    p.setPen(QPen(accent, max(1.5, s * 0.02)))
    p.setBrush(fill)
    p.drawPath(path)
    # abbreviation, clearly below the drawing
    font = QFont(theme.FONT_FAMILY.split(',')[0].strip())
    font.setPointSizeF(max(8.0, s * 0.10))
    font.setBold(True)
    p.setFont(font)
    p.setPen(QColor(theme.TEXT))
    p.drawText(QRectF(0, h * 0.76, w, h * 0.24), Qt.AlignHCenter | Qt.AlignVCenter, ABBREVIATIONS.get(label, label[:4].upper()))
    p.end()
    return pix
