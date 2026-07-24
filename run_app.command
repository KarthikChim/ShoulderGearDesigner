#!/bin/zsh

set -u

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$PROJECT_DIR/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "Project interpreter was not found:"
  echo "$PYTHON"
  echo
  echo "Create it and install requirements before running."
  read -k 1 "?Press any key to close."
  exit 1
fi

cd "$PROJECT_DIR"
exec "$PYTHON" main.py
