#!/usr/bin/env bash
# Set up a Python environment for std-t98-tools.
#
# Two interpreters, because the RF and audio halves have disjoint, heavy, and
# awkward-to-coinstall dependencies (see README):
#
#   system python  drives the RF backend, using the distro's GNU Radio/SoapySDR
#   env/           a venv for the audio/protocol/secret services
#
# This script only sets up env/ and checks that the RF side is usable; it never
# touches the system Python or installs system packages, so it needs no root.
# It is safe to re-run.
#
# Flags:
#   --with-secret   also install torch + safetensors (voice descrambling)
#   --with-dev      also install pytest
#   --help

set -euo pipefail
cd "$(dirname "$0")"

WITH_SECRET=0
WITH_DEV=0
for arg in "$@"; do
    case "$arg" in
        --with-secret) WITH_SECRET=1 ;;
        --with-dev) WITH_DEV=1 ;;
        --help|-h)
            sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "unknown option: $arg" >&2; exit 1 ;;
    esac
done

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!!  %s\033[0m\n' "$*"; }

# --- RF side: report, do not install (it is distro-provided) ----------------
say "Checking the RF stack (system Python)"
rf_ok=1
if python3 -c "from gnuradio import gr, soapy" 2>/dev/null; then
    echo "  GNU Radio + gnuradio.soapy: OK ($(python3 -c 'from gnuradio import gr; print(gr.version())'))"
else
    warn "GNU Radio (with gnuradio.soapy) not found for system python3."
    warn "Install it from your distro or radioconda; see README 動作要件."
    rf_ok=0
fi
if command -v SoapySDRUtil >/dev/null 2>&1; then
    echo "  SoapySDR: OK"
else
    warn "SoapySDRUtil not found; install soapysdr-tools and a device module."
    rf_ok=0
fi

# --- service venv -----------------------------------------------------------
say "Creating the service venv (env/)"
# --system-site-packages so the services can see the distro numpy/gnuradio if
# they need to; the heavy wheels still install into env/.
python3 -m venv --system-site-packages env
env/bin/pip install --quiet --upgrade pip

say "Installing the audio stack"
env/bin/pip install --quiet -r requirements-audio.txt

say "Building pyambelib (AMBE decoder)"
if env/bin/python -c "import pyambelib" 2>/dev/null; then
    echo "  already installed"
else
    if ! command -v cc >/dev/null 2>&1 && ! command -v gcc >/dev/null 2>&1; then
        warn "no C compiler found; install build-essential, then re-run."
    fi
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT
    git clone --depth 1 --recurse-submodules -q \
        https://github.com/tallcat4/pyambelib "$tmp/pyambelib"
    env/bin/pip install --quiet "$tmp/pyambelib"
fi

if [[ "$WITH_SECRET" == 1 ]]; then
    say "Installing the secret (voice descrambling) stack -- torch, this is large"
    env/bin/pip install --quiet -r requirements-secret.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu
fi

if [[ "$WITH_DEV" == 1 ]]; then
    say "Installing dev tools"
    env/bin/pip install --quiet -r requirements-dev.txt
fi

# --- verify -----------------------------------------------------------------
say "Verifying the service venv"
env/bin/python - <<'PY'
import importlib
ok = True
for mod, label in [("numpy", "numpy"), ("sounddevice", "sounddevice (PortAudio)"),
                   ("pyambelib", "pyambelib")]:
    try:
        importlib.import_module(mod)
        print(f"  {label}: OK")
    except Exception as e:
        print(f"  {label}: FAIL -- {e}")
        ok = False
for mod in ("torch", "safetensors"):
    try:
        importlib.import_module(mod)
        print(f"  {mod}: OK (secret service available)")
    except Exception:
        print(f"  {mod}: not installed (secret service disabled; --with-secret to add)")
raise SystemExit(0 if ok else 1)
PY

say "Done."
if [[ "${rf_ok}" == 0 ]]; then
    warn "The audio side is ready but the RF side is incomplete (see above)."
    warn "You can still work offline with --replay on a recording."
fi
echo "Next: python3 std_t98_multi_service_launcher.py   (see README 使い方)"
