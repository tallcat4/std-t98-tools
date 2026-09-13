# -*- coding: utf-8 -*-
"""Entry point: ``python -m app`` launches the desktop front-end."""

import sys
from pathlib import Path

from PyQt5 import QtCore, QtWidgets

from app.main_window import MainWindow

# Harmless lines that Qt's platform / input-method plugins print on some
# desktops (ibus/fcitx create a QSocketNotifier at startup; Wayland cannot honour
# a window-activation request). They come from the plugin, not from this app --
# we use no QSocketNotifier and never call requestActivate/activateWindow. Drop
# exactly these; pass every other Qt message through so real warnings stay.
_BENIGN = (
    "QSocketNotifier: Can only be used with threads started with QThread",
    "Wayland does not support QWindow::requestActivate()",
)


def _is_benign(message):
    return any(line in message for line in _BENIGN)


def _message_filter(mode, context, message):
    if _is_benign(message):
        return
    sys.stderr.write(message + "\n")


def main(argv=None):
    argv = list(sys.argv if argv is None else argv)
    QtCore.qInstallMessageHandler(_message_filter)
    app = QtWidgets.QApplication(argv)
    app.setApplicationName("STD-T98 Multi Receiver")
    window = MainWindow(repo_root=Path(__file__).resolve().parent.parent)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
