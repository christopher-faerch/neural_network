"""Gesture view: one card per model class; the predicted class lights up with a red outline."""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from app import assets, theme
from network import config as C

CARD_IMAGE = QSize(180, 180)


class GestureCard(QFrame):
    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.label = label
        self.setProperty("role", "card")
        self.setProperty("state", "idle")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(150, 190)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 10)
        layout.setSpacing(6)
        self.image = QLabel()
        self.image.setAlignment(Qt.AlignCenter)
        self.image.setPixmap(assets.gesture_pixmap(label, CARD_IMAGE))
        self.name = QLabel(assets.pretty_name(label))
        self.name.setProperty("role", "name")
        self.name.setAlignment(Qt.AlignCenter)
        self.name.setWordWrap(True)
        self.conf = QLabel("")
        self.conf.setProperty("role", "conf")
        self.conf.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image, 1)
        layout.addWidget(self.name)
        layout.addWidget(self.conf)

    def set_state(self, state: str) -> None:
        if self.property("state") != state:
            self.setProperty("state", state)
            theme.repolish(self)
            theme.repolish(self.name)
            theme.repolish(self.conf)

    def set_confidence(self, value: float | None) -> None:
        self.conf.setText("" if value is None else f"{100 * value:.0f} %")


class GestureView(QWidget):
    """Grid of GestureCards plus a headline with the current prediction."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        head = QHBoxLayout()
        self.headline = QLabel("No prediction")
        self.headline.setProperty("role", "heading")
        self.detail = QLabel("")
        self.detail.setProperty("role", "dim")
        head.addWidget(self.headline)
        head.addStretch(1)
        head.addWidget(self.detail)
        layout.addLayout(head)
        grid = QGridLayout()
        grid.setSpacing(12)
        self.cards: dict[str, GestureCard] = {}
        columns = 5
        for i, label in enumerate(C.GESTURES):
            card = GestureCard(label)
            self.cards[label] = card
            grid.addWidget(card, i // columns, i % columns)
        layout.addLayout(grid, 1)
        self.current: str | None = None

    def clear(self) -> None:
        self.current = None
        self.headline.setText("No prediction")
        self.detail.setText("")
        for card in self.cards.values():
            card.set_state("idle")
            card.set_confidence(None)

    def show_prediction(self, payload: dict | None) -> None:
        """payload: {"raw": pred, "smoothed": pred|None, ...}; the smoothed prediction (if any) is highlighted."""
        if not payload:
            self.clear()
            return
        raw, smoothed = payload["raw"], payload.get("smoothed")
        shown = smoothed or raw
        probs = shown["probabilities"]
        self.current = shown["class_name"]
        for label, card in self.cards.items():
            idx = C.GESTURES.index(label)
            card.set_confidence(float(probs[idx]))
            card.set_state("predicted" if label == self.current else "dim")
        self.headline.setText(f"{assets.pretty_name(self.current)}   {100 * shown['confidence']:.0f} %")
        parts = []
        if smoothed is not None:
            parts.append(f"smoothed over {payload.get('smooth_n', C.SMOOTH_WINDOWS)} windows; raw: "
                         f"{assets.pretty_name(raw['class_name'])} {100 * raw['confidence']:.0f} %")
        if payload.get("calibrated"):
            parts.append("calibrated model")
        if "end_sample" in payload and payload.get("fs"):
            parts.append(f"window ending at {payload['end_sample'] / payload['fs']:.1f} s")
        if "latency_ms" in payload:
            parts.append(f"{payload['latency_ms']:.0f} ms")
        self.detail.setText("  |  ".join(parts))
