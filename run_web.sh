#!/usr/bin/env bash
# One-click launcher for the web version (delegates to the PowerShell engine).
set -e
cd "$(dirname "$0")"
if command -v pwsh >/dev/null 2>&1; then ps=pwsh; else ps=powershell; fi
exec "$ps" -NoProfile -ExecutionPolicy Bypass -File ./run_web.ps1 "$@"
