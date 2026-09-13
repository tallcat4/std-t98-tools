#!/usr/bin/env bash
# Register the desktop front-end as an application, so it launches from the
# menu / app grid instead of `python -m app`. No root: it only writes into the
# per-user ~/.local/share tree, and is safe to re-run. `--uninstall` removes it.
#
# It bakes an absolute Exec into the .desktop file, picking a Python that can
# import PyQt5 (env/bin/python if setup.sh built it, else the system python3),
# so the entry works regardless of the current directory.

set -euo pipefail
cd "$(dirname "$0")"
REPO="$(pwd)"
APP_ID="std-t98-receiver"

DESKTOP_DIR="$HOME/.local/share/applications"
ICON_DIR="$HOME/.local/share/icons/hicolor/scalable/apps"
DESKTOP_FILE="$DESKTOP_DIR/$APP_ID.desktop"
ICON_FILE="$ICON_DIR/$APP_ID.svg"

refresh() {
    command -v update-desktop-database >/dev/null 2>&1 \
        && update-desktop-database "$DESKTOP_DIR" >/dev/null 2>&1 || true
    command -v gtk-update-icon-cache >/dev/null 2>&1 \
        && gtk-update-icon-cache -f -t "$HOME/.local/share/icons/hicolor" >/dev/null 2>&1 || true
}

if [[ "${1:-}" == "--uninstall" ]]; then
    rm -f "$DESKTOP_FILE" "$ICON_FILE"
    refresh
    echo "Removed $APP_ID desktop entry and icon."
    exit 0
fi
if [[ -n "${1:-}" ]]; then
    echo "usage: $0 [--uninstall]" >&2
    exit 1
fi

# Pick an interpreter that can import PyQt5.
GUI_PYTHON=""
for candidate in "$REPO/env/bin/python" "$(command -v python3 || true)" "$(command -v python || true)"; do
    [[ -n "$candidate" && -x "$candidate" ]] || continue
    if "$candidate" -c "import PyQt5" >/dev/null 2>&1; then
        GUI_PYTHON="$candidate"
        break
    fi
done
if [[ -z "$GUI_PYTHON" ]]; then
    echo "No Python with PyQt5 found. Run ./setup.sh, or install PyQt5 (e.g. the" >&2
    echo "distro's python3-pyqt5), then re-run this script." >&2
    exit 1
fi

mkdir -p "$DESKTOP_DIR" "$ICON_DIR"
cp "$REPO/app/icon.svg" "$ICON_FILE"

cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Version=1.0
Name=STD-T98 Multi Receiver
Comment=ARIB STD-T98 multi-channel SDR receiver
Exec=$GUI_PYTHON "$REPO/std_t98_gui.py"
Path=$REPO
Icon=$APP_ID
Terminal=false
Categories=HamRadio;Network;Utility;
Keywords=SDR;radio;STD-T98;receiver;
StartupNotify=true
EOF

# Some desktops only trust .desktop files that are marked executable.
chmod +x "$DESKTOP_FILE"
refresh

echo "Installed:"
echo "  $DESKTOP_FILE"
echo "  $ICON_FILE"
echo "  Python: $GUI_PYTHON"
echo
echo "Launch it from the application menu (search \"STD-T98\")."
echo "Remove it with: $0 --uninstall"
