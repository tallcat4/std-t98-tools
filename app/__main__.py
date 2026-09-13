# -*- coding: utf-8 -*-
"""Entry point: ``python -m app`` launches the desktop front-end."""

import sys
from pathlib import Path

from PyQt5 import QtCore, QtWidgets

from app.main_window import MainWindow

# Qt's platform / input-method plugins (ibus, fcitx, ...) create a
# QSocketNotifier during startup and print this exact line on some desktops. It
# comes from the plugin, not from this app (we touch no sockets before Start and
# use no QSocketNotifier), and it is harmless. Drop just that line; pass every
# other Qt message through so real warnings stay visible.
_BENIGN = "QSocketNotifier: Can only be used with threads started with QThread"


def _message_filter(mode, context, message):
    if _BENIGN in message:
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
