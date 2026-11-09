#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

venv="$PWD/.venv"
python="$venv/bin/python"
stamp="$venv/.filmclear-requirements"

if [[ ! -x "$python" ]]; then
    python3 -m venv "$venv"
    rm -f "$stamp"
fi

requirements_checksum="$(cksum < requirements.txt)"
if [[ ! -f "$stamp" ]] || [[ "$(cat "$stamp")" != "$requirements_checksum" ]]; then
    "$python" -m pip install -r requirements.txt
    printf '%s\n' "$requirements_checksum" > "$stamp"
fi

if [[ "${1:-}" == "--test" ]]; then
    shift
    exec "$python" -m unittest discover -s tests "$@"
fi

if [[ "$(uname -s)" == Darwin ]]; then
    exec "$python" tools/macos_launch.py "$@"
fi

exec "$python" main.py "$@"
