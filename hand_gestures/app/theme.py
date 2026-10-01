"""Colours and the Qt stylesheet: near-black background, charcoal panels, turquoise accent, red for the
predicted gesture. Widgets never set colours themselves; they set a dynamic property (`role`, `state`)
and the stylesheet does the rest."""
from __future__ import annotations

BG = "#141618"            # window background (almost black)
BG_PANEL = "#1e2124"      # panels
BG_CARD = "#272b2f"       # cards, inputs
BG_HOVER = "#2f343a"
BORDER = "#363b41"
TEXT = "#e9ecef"
TEXT_DIM = "#8f979e"
ACCENT = "#2ec4b6"        # turquoise
ACCENT_DIM = "#1b7f76"
ACCENT_TEXT = "#0b1a19"   # text on an accent-filled control
PREDICTED = "#ff2e2e"     # red outline of the predicted gesture
WARN = "#f2b134"
ERROR = "#e5484d"
OK = "#39c37d"
LIVE = "#ff5c5c"          # LIVE HARDWARE badge
REPLAY = "#f2b134"        # SIMULATED / REPLAY badge
FILE = ACCENT             # FILE badge

# same order and hues as network.plots.CHANNEL_COLORS (copied: importing plots would pull in matplotlib)
CHANNEL_COLORS = ["#4da3ff", "#ff8c42", "#2ec4b6", "#ffd166"]

FONT_FAMILY = "Segoe UI, Inter, Helvetica Neue, Arial, sans-serif"

QSS = f"""
* {{ font-family: {FONT_FAMILY}; font-size: 13px; }}
QMainWindow, QWidget {{ background: {BG}; color: {TEXT}; }}
QToolTip {{ background: {BG_CARD}; color: {TEXT}; border: 1px solid {BORDER}; padding: 4px 6px; }}

QFrame[role="panel"] {{ background: {BG_PANEL}; border: 1px solid {BORDER}; border-radius: 10px; }}
QFrame[role="statusbar"] {{ background: {BG_PANEL}; border-top: 1px solid {BORDER}; }}
QLabel {{ background: transparent; border: none; }}
QLabel[role="title"] {{ font-size: 11px; font-weight: 600; letter-spacing: 1.2px; color: {ACCENT}; }}
QLabel[role="heading"] {{ font-size: 18px; font-weight: 600; color: {TEXT}; }}
QLabel[role="dim"] {{ color: {TEXT_DIM}; }}
QLabel[role="value"] {{ color: {TEXT}; font-weight: 600; }}
QLabel[role="ok"] {{ color: {OK}; font-weight: 600; }}
QLabel[role="warn"] {{ color: {WARN}; }}
QLabel[role="error"] {{ color: {ERROR}; }}
QLabel[role="badge-live"] {{ background: {LIVE}; color: #1a0606; font-weight: 700; border-radius: 6px; padding: 3px 10px; letter-spacing: 1px; }}
QLabel[role="badge-replay"] {{ background: {REPLAY}; color: #1d1400; font-weight: 700; border-radius: 6px; padding: 3px 10px; letter-spacing: 1px; }}
QLabel[role="badge-file"] {{ background: {ACCENT}; color: {ACCENT_TEXT}; font-weight: 700; border-radius: 6px; padding: 3px 10px; letter-spacing: 1px; }}
QLabel[role="badge-off"] {{ background: {BG_CARD}; color: {TEXT_DIM}; font-weight: 700; border-radius: 6px; padding: 3px 10px; letter-spacing: 1px; border: 1px solid {BORDER}; }}

QPushButton {{ background: {BG_CARD}; color: {TEXT}; border: 1px solid {BORDER}; border-radius: 7px; padding: 6px 12px; }}
QPushButton:hover {{ background: {BG_HOVER}; border-color: {ACCENT_DIM}; }}
QPushButton:pressed {{ background: {ACCENT_DIM}; color: {ACCENT_TEXT}; }}
QPushButton:checked {{ background: {ACCENT}; color: {ACCENT_TEXT}; border-color: {ACCENT}; font-weight: 600; }}
QPushButton:disabled {{ color: {TEXT_DIM}; background: {BG_PANEL}; border-color: {BORDER}; }}
QPushButton[role="primary"] {{ background: {ACCENT}; color: {ACCENT_TEXT}; border-color: {ACCENT}; font-weight: 600; }}
QPushButton[role="primary"]:hover {{ background: #45d4c6; }}
QPushButton[role="primary"]:disabled {{ background: {BG_PANEL}; color: {TEXT_DIM}; border-color: {BORDER}; }}
QPushButton[role="danger"] {{ border-color: {ERROR}; color: {ERROR}; }}
QPushButton[role="danger"]:hover {{ background: {ERROR}; color: #ffffff; }}
QPushButton[role="danger"]:disabled {{ color: {TEXT_DIM}; border-color: {BORDER}; }}

QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{ background: {BG_CARD}; color: {TEXT}; border: 1px solid {BORDER}; border-radius: 6px; padding: 4px 8px; selection-background-color: {ACCENT_DIM}; }}
QComboBox:hover, QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover {{ border-color: {ACCENT_DIM}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{ background: {BG_CARD}; color: {TEXT}; border: 1px solid {BORDER}; selection-background-color: {ACCENT_DIM}; outline: none; }}
QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border: 1px solid {BORDER}; border-radius: 4px; background: {BG_CARD}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}

QProgressBar {{ background: {BG_CARD}; border: 1px solid {BORDER}; border-radius: 6px; text-align: center; color: {TEXT}; height: 14px; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 5px; }}
QSlider::groove:horizontal {{ height: 4px; background: {BG_CARD}; border-radius: 2px; }}
QSlider::handle:horizontal {{ width: 14px; height: 14px; margin: -5px 0; background: {ACCENT}; border-radius: 7px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT_DIM}; border-radius: 2px; }}
QScrollBar:horizontal {{ height: 10px; background: {BG_PANEL}; border: none; }}
QScrollBar::handle:horizontal {{ background: {BORDER}; border-radius: 5px; min-width: 30px; }}
QScrollBar::handle:horizontal:hover {{ background: {ACCENT_DIM}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar:vertical {{ width: 10px; background: {BG_PANEL}; border: none; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 5px; min-height: 30px; }}
QScrollArea {{ border: none; }}
QSplitter::handle {{ background: {BORDER}; }}
QMessageBox {{ background: {BG_PANEL}; }}

GestureCard {{ background: {BG_CARD}; border: 2px solid {BORDER}; border-radius: 12px; }}
GestureCard[state="predicted"] {{ border: 3px solid {PREDICTED}; background: #2d2a2e; }}
GestureCard[state="dim"] {{ border: 2px solid {BORDER}; background: #202326; }}
GestureCard QLabel[role="name"] {{ font-size: 14px; font-weight: 600; color: {TEXT}; }}
GestureCard[state="dim"] QLabel[role="name"] {{ color: {TEXT_DIM}; }}
GestureCard QLabel[role="conf"] {{ font-size: 12px; color: {TEXT_DIM}; }}
GestureCard[state="predicted"] QLabel[role="conf"] {{ color: {PREDICTED}; font-weight: 700; font-size: 14px; }}
"""


def apply_theme(app) -> None:
    """Fusion style + the stylesheet above, and matching pyqtgraph defaults."""
    app.setStyle("Fusion")
    app.setStyleSheet(QSS)
    try:
        import pyqtgraph as pg
        pg.setConfigOptions(background=BG_PANEL, foreground=TEXT_DIM, antialias=False)
    except ImportError:               # plotting is optional for the smoke tests
        pass


def repolish(widget) -> None:
    """Re-apply the stylesheet after a dynamic property changed (Qt does not do this automatically)."""
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()
