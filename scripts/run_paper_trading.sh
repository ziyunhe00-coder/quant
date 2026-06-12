#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG_PATH="${1:-config/paper_trading_config.json}"
if [[ ! -f "$CONFIG_PATH" ]]; then
  CONFIG_PATH="config/paper_trading_config.example.json"
fi

mkdir -p logs paper_trading

if [[ ! -x ".venv/bin/python" ]]; then
  python3 -m venv .venv
fi

DEPS_MARKER=".venv/.paper_trading_deps_installed"
if [[ ! -f "$DEPS_MARKER" || "requirements.txt" -nt "$DEPS_MARKER" ]]; then
  ".venv/bin/python" -m pip install -q -r requirements.txt
  touch "$DEPS_MARKER"
fi
".venv/bin/python" src/paper_trading_daily.py --config "$CONFIG_PATH" 2>&1 | tee -a "logs/paper_trading_$(date +%Y%m%d).log"
