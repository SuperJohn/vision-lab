#!/bin/zsh
# Double-click in Finder to start Vision Lab and open it in your browser.
cd "$(dirname "$0")"
URL="http://localhost:${PORT:-8765}"
if curl -s -o /dev/null "$URL/api/state"; then
  open "$URL"   # already running
  exit 0
fi
(for i in {1..60}; do curl -s -o /dev/null "$URL/api/state" && open "$URL" && break; sleep 1; done) &
./run.sh
