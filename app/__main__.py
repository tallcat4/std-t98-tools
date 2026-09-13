# -*- coding: utf-8 -*-
"""Entry point: ``python -m app`` launches the desktop front-end."""

import sys
from pathlib import Path

from PyQt5 import QtWidgets

from app.main_window import MainWindow


def main(argv=None):
    argv = list(sys.argv if argv is None else argv)
    app = QtWidgets.QApplication(argv)
    app.setApplicationName("STD-T98 Multi Receiver")
    window = MainWindow(repo_root=Path(__file__).resolve().parent.parent)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
