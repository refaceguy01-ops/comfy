#!/bin/sh
# ============================================================
#  ComfyUI model setup — just double-click this file.
#  It quietly fetches its own tools; nothing to install first.
# ============================================================
cd "$(dirname "$0")" || exit 1
echo
echo "  Starting the ComfyUI setup wizard..."
echo "  (a browser window will open in a moment — leave THIS window open)"
echo

UV="$(command -v uv || true)"
if [ -z "$UV" ]; then
    for cand in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
        [ -x "$cand" ] && UV="$cand" && break
    done
fi
if [ -z "$UV" ]; then
    echo "  First run: fetching a small helper tool (about 20 MB)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
    UV="$HOME/.local/bin/uv"
fi
if [ ! -x "$UV" ]; then
    echo "  Could not fetch the helper tool. Are you connected to the internet?"
    echo "  If this keeps happening, see TROUBLESHOOTING.md."
    read -r _
    exit 1
fi

"$UV" run --python 3.12 provision.py wizard || {
    echo
    echo "  Setup closed with a problem. Check the logs folder for details,"
    echo "  or just run Setup again — downloads continue where they left off."
    read -r _
}
