#!/bin/zsh
# HOU-88 publish gate: run before making the repo public. Exits non-zero on any finding.
# Usage: [VLAB_DATA=path/to/data] scripts/publish_gate.sh   (needs: brew install gitleaks)
set -u
cd "$(dirname "$0")/.."
fail=0
echo "== gitleaks: full history (all refs)"
gitleaks git --log-opts="--all" --redact --no-banner . || fail=1
echo "== gitleaks: working tree"
gitleaks dir --redact --no-banner . || fail=1
echo "== PII regex scan: tracked files (counts only)"
DATA=${VLAB_DATA:-data}
if [ ! -f "$DATA/redaction_terms.txt" ] || [ ! -f "$DATA/redaction_title_terms.txt" ]; then
  echo "  no local term lists in $DATA (set VLAB_DATA=<live repo>/data); failing closed"; fail=1
fi
TERMS=(${(f)"$(cat "$DATA/redaction_terms.txt" "$DATA/redaction_title_terms.txt" 2>/dev/null | grep -v '^#' | sed '/^$/d')"})
PATS=('[0-9]{3}[-. )]+[0-9]{3}[-. ][0-9]{4}' '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[a-z]{2,}' '/Users/[a-z]' "${TERMS[@]}")
for p in "${PATS[@]}"; do
  n=$(git grep -I -i -c -E -e "$p" -- . ':!*.db' ':!scripts/publish_gate.sh' 2>/dev/null | awk -F: '{s+=$NF} END {print s+0}')
  if [ "$n" -gt 0 ]; then echo "  FOUND $n hit(s) for a ${#p}-char pattern"; fail=1; fi
done
echo "== PII regex scan: history (local terms only, counts only)"
for p in "${TERMS[@]}"; do
  n=$(git log --all -i -G"$p" --format=%h 2>/dev/null | wc -l | tr -d ' ')
  if [ "$n" -gt 0 ]; then echo "  history: $n commit(s) touch a local term (${#p} chars)"; fail=1; fi
done
[ $fail -eq 0 ] && echo "PUBLISH GATE: PASS" || echo "PUBLISH GATE: FAIL"
exit $fail
