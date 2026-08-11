#!/usr/bin/env bash
set -euo pipefail

BASE="/home/exouser/course-ai/coursebot"
export CHROMA_DIR="/home/exouser/course-ai/chroma_db"
PY="$BASE/venv/bin/python"

# sanity check
if [[ ! -x "$PY" ]]; then
  echo "[error] venv python not found at: $PY"
  exit 1
fi

RAW="$BASE/course_docs/notebooklm_raw"
CLEAN="$BASE/course_docs/notebooklm_clean"
PDFS="$BASE/course_docs/pdfs"

mkdir -p "$RAW" "$CLEAN" "$PDFS" "$BASE/logs"

echo "==== $(date) ===="
echo "[info] using python: $($PY -c 'import sys; print(sys.executable)')"
echo "[info] CHROMA_DIR=${CHROMA_DIR:-<UNSET>}"

# NOTE: cleaner is under ingest/
$PY "$BASE/ingest/clean_notebooklm.py" --in_dir "$RAW" --out_dir "$CLEAN"
$PY "$BASE/ingest/ingest_to_chroma.py" --notes_dir "$CLEAN" --pdf_dir "$PDFS"

echo "[done]"