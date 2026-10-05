#!/usr/bin/env bash
# StreamVerse — one command to start.
#   ./run.sh
# Then open http://127.0.0.1:8000

set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "Creating virtual environment…"
  python3 -m venv .venv
  ./.venv/bin/pip install --quiet --upgrade pip
  ./.venv/bin/pip install --quiet -r backend/requirements.txt
fi

if [ ! -f backend/model/streamverse_bundle.pkl ]; then
  echo
  echo "  backend/model/streamverse_bundle.pkl is missing."
  echo "  Download it from Colab (streamverse_model/streamverse_bundle.pkl)"
  echo "  and put it in backend/model/ — then run this again."
  echo
  exit 1
fi

cd backend
exec ../.venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000 --reload
