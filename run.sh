#!/bin/zsh
# Start Vision Lab on http://localhost:8765 (creates the virtualenv on first run).
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
fi
K=weights/kokoro; U=https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0
if [ ! -f $K/voices-v1.0.bin ]; then
  mkdir -p $K && curl -sSL -o $K/kokoro-v1.0.onnx $U/kokoro-v1.0.onnx && curl -sSL -o $K/voices-v1.0.bin $U/voices-v1.0.bin
fi
exec .venv/bin/python -m uvicorn vision_lab.server:app --host 127.0.0.1 --port "${PORT:-8765}"
