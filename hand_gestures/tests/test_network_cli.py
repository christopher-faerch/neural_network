"""The network package works headless: imports without Qt, and `python -m network.main` dispatches."""
import subprocess
import sys

import pytest

from network import main as network_main


def test_network_imports_without_qt():
    code = ("import sys, network.main, network.model, network.data_loader, network.inference, "
            "network.streaming, network.recording, network.predict, network.evaluate, network.cross_validate; "
            "print('PySide6' in sys.modules or 'pyqtgraph' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_main_without_command_prints_usage(capsys):
    assert network_main.main([]) == 2
    out = capsys.readouterr().out
    for command in ("train", "evaluate", "predict", "cross-validate", "summary"):
        assert command in out


def test_main_unknown_command(capsys):
    assert network_main.main(["frobnicate"]) == 2
    assert "unknown command" in capsys.readouterr().err


def test_main_passes_arguments_to_the_module():
    with pytest.raises(SystemExit) as exc:
        network_main.main(["predict", "--help"])          # argparse --help exits with 0 after printing
    assert exc.value.code == 0


def test_summary_command_runs(capsys):
    assert network_main.main(["summary"]) == 0
    out = capsys.readouterr().out
    assert "total trainable parameters: 63,274" in out
    assert "[1, 4, 2000]" in out
