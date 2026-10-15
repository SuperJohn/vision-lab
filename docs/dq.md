# Data-quality checks (HOU-85)

Run: `python -m vision_lab.dq --db data/vlab.db` (read-only; add `--json` for machine output). Definitions are in the `vision_lab/dq.py` docstring; tests run on the synthetic fixture (`tests/test_dq.py`).

## Aggregate results on the live DB (2026-10-14, 8 PM PT)

Aggregate counts and rates only; no row content.

conf threshold 0.5 · person IoU 0.7 · track window 2.0s · embedding cos 0.95

| check | metric | before threshold | after threshold |
|---|---|---|---|
| couch_fp | couch detections | 2012 | 1300 |
| couch_fp | flagged (screen/bed/low-conf) | 763 | 18 |
| couch_fp |   on screen capture | 18 | 15 |
| couch_fp |   overlaps bed | 185 | 3 |
| couch_fp |   below conf | 712 | 0 |
| couch_fp | FP rate | 0.3792 | 0.0138 |
| dup_person | frames with a person | 1075 | 901 |
| dup_person | frames with duplicate box | 1 | 0 |
| dup_person | duplicate-frame rate | 0.0009 | 0.0 |
| dup_person | person tracks | 3608 | - |
| dup_person | candidate duplicate track pairs | 483 | - |
| dup_person |   confirmed / rejected / unverified | 94 / 7 / 382 | - |
| dup_person | candidate rate (per track) | 0.1339 | - |
| dup_person | embedding coverage | 0.1904 | - |



## Reading the numbers



- **Couch false positives:** 38% of raw couch detections are flagged (mostly low confidence). A 0.5 confidence floor removes 35% of couch detections and cuts the flagged rate to 1.4%; the remainder are couches on screen captures (15) and couch boxes on a bed (3).

- **Duplicate person boxes in one frame** are essentially absent (1 of 1,075 frames), so NMS is fine.

- **Duplicate person tracks** are the real issue: 13% of person tracks have another person track starting within 2 s and overlapping. Only 19% of tracks have a keyframe embedding; of the pairs that could be checked, 94 of 101 (93%) look like the same person (cos >= 0.95). 382 pairs are unverified. Likely cause: ByteTrack re-assigning IDs after brief occlusion; worth tuning track buffer / re-ID before trusting person counts.

- Screen captures are excluded from detection-based metrics in future work: any detection there is not a physical object.
