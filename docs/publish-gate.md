# Publish gate report (HOU-88)

Run 2026-10-14 ~8:05 PM PT on branch `hou-88-publish-gate` (all refs: 22 commits). Findings are redacted: counts and file
names only. Re-run any time with `VLAB_DATA=<repo>/data scripts/publish_gate.sh` (exits non-zero on any finding).

## Tools
- `gitleaks` 8.30.1 (`brew install gitleaks`): `gitleaks git --log-opts=--all` (full history) and `gitleaks dir` (tree).
- Regex PII scan (`git grep` on the tree, `git log -G` across all refs): phone numbers, emails, street addresses,
  `/Users/<name>` paths, and every term in the local, untracked lists `data/redaction_terms.txt` (names) and
  `data/redaction_title_terms.txt` (places, employer, ZIPs).

## Results

| Scan | Tree | History |
|---|---|---|
| gitleaks secrets | 0 | 0 |
| Phone / email / street address / home path | 0 | 0 |
| Personal names (child, family) | 0 | **yes**: `vision_lab/planloop.py` (config keys + function named after a child, 5 commits, since renamed to `kid_*`), `vision_lab/redaction_pack.py` (denylist regex with family names, 1 commit) |
| Places / employer / ZIP prefix | **1 file, fixed on this branch** | **yes**: `vision_lab/redaction_pack.py` (title denylist + ZIP regex, 12 commits) |
| Synthetic fixture `tests/fixtures/vlab_fixture.db` | 0 | n/a |
| `macapp/*` `com.houghtonlabs` bundle ID, LICENSE copyright | intended (author identity) | intended |

The gate currently **FAILS** on history: 10 local terms appear in 1–3 commits each.

## Tree fix (this branch)
`vision_lab/redaction_pack.py` no longer contains the place/employer/ZIP terms. They now load from
`data/redaction_title_terms.txt` (git-ignored; created locally with the same terms so filtering behaviour is unchanged),
deny matching window titles, and are masked as `[LOCAL_TERM]` in text. Tested in `tests/redaction_terms.py` with a synthetic term.

## History
The public repository is published as a single squashed snapshot of the cleaned tree, so no earlier
history is included. Run the gate again before every publish.
