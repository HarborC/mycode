#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rerun_bin="${HOME}/.local/bin/rerun"
default_args=(--new)

if command -v rerun >/dev/null 2>&1; then
  rerun_bin="$(command -v rerun)"
fi

if [[ ! -x "$rerun_bin" ]]; then
  echo "Rerun viewer not found."
  echo "Install it with: pip3 install --user rerun-sdk"
  exit 1
fi

mapfile -t rrd_files < <(find "$root_dir" -maxdepth 2 -name 'clip.rrd' | sort)

if [[ ${#rrd_files[@]} -eq 0 ]]; then
  echo "No clip.rrd files found under $root_dir"
  exit 1
fi

if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" && -z "${WAYLAND_SOCKET:-}" ]]; then
  default_args=(--web-viewer --bind 127.0.0.1 --web-viewer-port 9091)
  echo "No desktop display detected. Starting Rerun Web Viewer at http://127.0.0.1:9091"
fi

exec "$rerun_bin" "${default_args[@]}" "$@" "${rrd_files[@]}"
