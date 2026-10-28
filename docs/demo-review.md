# HOU-153 demo site review sheet

All media are keyframes from the lab camera (no people, no faces, no screens, no outdoor views). Screened by gemma4:26b, then by eye on a contact sheet. Boxes drawn for detections with conf >= 0.5.

| asset | source keyframe | label | why chosen |
|---|---|---|---|
| frame0_annotated.jpg | 1791680242971.jpg | empty | VLM score 7/10; shows chair, couch, refrigerator |
| frame1_annotated.jpg | 1791766184646.jpg | empty | VLM score 7/10; shows couch, refrigerator |
| frame2_annotated.jpg | 1791675136883.jpg | kitchen | VLM score 5/10; shows chair, couch, dog, refrigerator |
| frame3_annotated.jpg | 1791675066779.jpg | dog_only | VLM score 7/10; shows chair, couch, dog, refrigerator |
| frame4_annotated.jpg | 1791684335673.jpg | dog_only | VLM score 5/10; shows chair, couch, dog, refrigerator |
| frame5_annotated.jpg | 1791673248934.jpg | dog_only | VLM score 7/10; shows chair, couch, dog, refrigerator |
| frame6_annotated.jpg | 1791775560498.jpg | at_desk | VLM score 7/10; shows chair, couch, refrigerator |
| detections.mp4 / .webm | the 7 frames above | - | 9 s slideshow of annotated frames |

## Rejected by VLM screen

8 of 36 candidates flagged for people/faces (despite no YOLO person box) and excluded; the rest of the rejections were low showcase scores.

## Borderline (John to decide)

- All frames show the interior layout of the home (living room/kitchen). No windows to the outside, addresses or documents are legible at 960 px, but the layout itself is recognizable to anyone who has visited.
- The dog appears in frame2, frame3, frame4 and frame5 (and the clip).
- A wall whiteboard by the front door and fridge/counter items are visible at low resolution; nothing legible was seen at 960 px, but check the whiteboard before sharing.
- Not used: data/recordings/rec-20261010-144824.mp4 and data/screens (753 desktop screenshots) because of people or personal screen content; no UI screenshots of the live dashboard were taken because it shows the live camera.
