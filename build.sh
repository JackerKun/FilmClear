#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

case "$(uname -s)" in
    Darwin) platform=macos ;;
    Linux) platform=linux ;;
    *) echo "Use build.bat on Windows." >&2; exit 1 ;;
esac

bootstrap="${FILMCLEAR_PYTHON:-python3}"
if [[ -z "${FILMCLEAR_PYTHON:-}" && -x .venv/bin/python ]]; then
    bootstrap=.venv/bin/python
fi
if ! command -v "$bootstrap" >/dev/null 2>&1; then
    echo "Install Python 3.10+ or set FILMCLEAR_PYTHON to its executable path." >&2
    exit 1
fi

arch="$("$bootstrap" -c 'import platform; print(platform.machine())')"
venv=".venv-build/$platform-$arch"
python="$venv/bin/python"
work="build/$platform-$arch"
dist="dist/$platform"

if [[ ! -x "$python" ]]; then
    "$bootstrap" -m venv "$venv"
fi
"$python" -m pip --disable-pip-version-check install -r requirements-build.txt

mkdir -p "$work"
export PYINSTALLER_CONFIG_DIR="$PWD/$work/cache"
options=(--console)
if [[ "$platform" == macos ]]; then
    options=(--windowed --osx-bundle-identifier com.filmclear.app --icon "$PWD/assets/filmclear.icns")
fi
"$python" -m PyInstaller \
    --name FilmClear --onedir --noconfirm --noupx \
    --distpath "$dist" --workpath "$work/work" --specpath "$work" \
    --collect-data cv2.data \
    --add-data "$PWD/assets/filmclear.png:assets" \
    --add-data "$PWD/models/face_detection_yunet_2022mar.onnx:models" \
    --add-data "$PWD/models/face_detection_yunet_2023mar.onnx:models" \
    --add-data "$PWD/models/YuNet-LICENSE.txt:models" \
    "${options[@]}" main.py

if [[ "$platform" == macos ]]; then
    echo "Built: $PWD/$dist/FilmClear.app"
else
    echo "Built: $PWD/$dist/FilmClear/FilmClear"
fi
