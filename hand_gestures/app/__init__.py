"""Desktop application (PySide6) for live and offline EMG hand-gesture recognition.

The app owns Qt, acquisition and visual state; everything about the network lives in the `network`
package and is reached only through `network.inference`, `network.recording` and `network.streaming`.
Run with `python -m app.main`.
"""
