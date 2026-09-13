#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convenience launcher for the desktop front-end (same as ``python -m app``)."""

from app.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
