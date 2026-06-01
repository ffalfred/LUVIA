#!/usr/bin/env bash
# Build LUVIA as a Linux AppImage.
#
# Usage:
#   packaging/build_appimage.sh           # builds the CUDA variant (~4 GB)
#   packaging/build_appimage.sh --cpu     # builds the CPU-only variant (~1.5 GB)
#   packaging/build_appimage.sh --cuda    # same as default
#
# The two variants live in parallel conda envs (luvia_py311 / luvia_py311_cpu)
# because PyTorch CPU and CUDA wheels can't coexist in one env. Output AppImage
# is suffixed with the variant name.

set -euo pipefail

VARIANT="cuda"
for arg in "$@"; do
  case "$arg" in
    --cpu)  VARIANT="cpu" ;;
    --cuda) VARIANT="cuda" ;;
    *) echo "Unknown argument: $arg"; echo "Usage: $0 [--cpu|--cuda]"; exit 1 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ "$VARIANT" = "cpu" ]; then
  ENV_PATH="/home/alfredff/Bio_Software/miniconda3/envs/luvia_py311_cpu"
  APPIMAGE_NAME="LUVIA-cpu-x86_64.AppImage"
else
  ENV_PATH="/home/alfredff/Bio_Software/miniconda3/envs/luvia_py311"
  APPIMAGE_NAME="LUVIA-cuda-x86_64.AppImage"
fi

PYINSTALLER="${ENV_PATH}/bin/pyinstaller"
DIST_DIR="${REPO_ROOT}/dist"
APPDIR="${REPO_ROOT}/build/LUVIA-${VARIANT}.AppDir"
APPIMAGETOOL="${REPO_ROOT}/build/appimagetool-x86_64.AppImage"
PYINSTALLER_OUT="${DIST_DIR}/luvia"

if [ ! -x "$PYINSTALLER" ]; then
  echo "PyInstaller not found at $PYINSTALLER"
  echo "Expected env: $ENV_PATH"
  exit 1
fi

echo "==> Building ${VARIANT} variant"
echo "==> Env:           $ENV_PATH"
echo "==> Output:        $APPIMAGE_NAME"

echo "==> Running PyInstaller"
"$PYINSTALLER" --noconfirm --clean packaging/LUVIA.spec

echo "==> Preparing AppDir"
rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin"
cp -a "$PYINSTALLER_OUT/." "$APPDIR/usr/bin/"

cp "${REPO_ROOT}/src/luvia_gui/gifs/signal-2025-08-25-003555_003.png" "$APPDIR/luvia.png"

cat > "$APPDIR/luvia.desktop" <<EOF
[Desktop Entry]
Name=LUVIA (${VARIANT})
Exec=luvia
Icon=luvia
Type=Application
Categories=Graphics;Science;
EOF

cat > "$APPDIR/AppRun" <<'EOF'
#!/bin/bash
HERE="$(dirname "$(readlink -f "${0}")")"
export PATH="${HERE}/usr/bin:${PATH}"
exec "${HERE}/usr/bin/luvia" "$@"
EOF
chmod +x "$APPDIR/AppRun"

echo "==> Fetching appimagetool if missing"
mkdir -p "$(dirname "$APPIMAGETOOL")"
if [ ! -x "$APPIMAGETOOL" ]; then
  wget -q -O "$APPIMAGETOOL" \
    "https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage"
  chmod +x "$APPIMAGETOOL"
fi

echo "==> Building AppImage"
ARCH=x86_64 "$APPIMAGETOOL" "$APPDIR" "${DIST_DIR}/${APPIMAGE_NAME}"

echo "==> Done: ${DIST_DIR}/${APPIMAGE_NAME}"
ls -lh "${DIST_DIR}/${APPIMAGE_NAME}"
