# Demo data (Vision Lab)

Synthetic / privacy-screened sample data for the Cloudflare Pages showcase and for running Vision Lab tooling without a
live camera. **All fictional:** a made-up one-week timeline (Mon 2026-10-19 to Sun 2026-10-25, America/Los_Angeles) for
"Demo Apartment" with one resident ("Resident A") and a small dog. No real camera captures, voices, people or homes.

| File | What it is | Rows |
|---|---|---|
| `data.json` | Public showcase aggregates (stress test, label/kind counts, gallery captions). Unchanged; the Pages copy is `site/assets/data.json`. | 7 gallery assets |
| `config.json` | `zones` (couch, desk, kitchen, dog bed, front-door line; normalised polygon points) and `rules` (same shape as `config['rules']`). | 5 zones, 5 rules |
| `events.json` | Unified timeline in the `events` table shape (`ts`, `kind`, `modality`, `track_id`, `cls`, `keyframe_id`, `rule_id`, `vlm_text`, `data`, `text`, `color`): track, scene, zone, line, posture, rule, activity_slice, sound, voice, capture_paused/resumed. | 1,124 |
| `presence.json` | Open/close spans per track (`presence` table). | 197 |
| `keyframes.json` | Keyframe captions and labels (`keyframes` table; `path` points at placeholder names, `embedding` is null). | 184 |
| `batch_annotations.json` | Slow-loop output per keyframe: caption, label, detections `[{cls, conf, box}]` (normalised), `gemma_ms`, `detect_ms`. | 184 |
| `utterances.json` | Voice turns (user + assistant halves), each linked to the nearest keyframe, with latency breakdowns. | 24 (12 turns) |
| `daily_metrics.json` | Per-day roll-ups: events by kind, activity minutes by label, desk/couch hours, VLM p50/p95, privacy pauses. | 7 days |
| `plan_blocks.json` | Planloop nudges ("stand up and stretch") fired by the 90-minute desk rule. | 17 |
| `manifest.json` | Row counts, byte sizes and sha256 for every generated file. | |

Regenerate (seeded, byte-identical for the same seed): `python3 demo/generate_data.py`. It never touches `data.json` or `site/`.

**Local override.** Point tooling at your own data on the Studio with `HL_DATASET_URI`, e.g.
`HL_DATASET_URI=file://…/demo/data/data.json` (this folder) or a path to your local export; real data stays outside git.

Do not put real camera captures, family media, or PII here. Media under `site/assets/` was VLM-screened and hand-checked before commit.
