# RelateAnything with YOLO

The normal `python -m pipeline.main_loop` command now runs RelateAnything on
YOLO/ByteTrack boxes. It uses the released `maelic/relsgg-vits16plus` checkpoint;
its visual stage also loads YOLOv8n-seg for actual instance masks. Chair, laptop and cell-phone classes were
added to the existing detector class list, keeping the previous classes.

The separate `RelateAnything - segmentation and relations` window shows the
last analyzed camera frame with colored instance masks, object labels, and
labelled relation arrows. Results remain aligned to the exact image analyzed.
Segment detections match existing track IDs by class and box overlap;
unmatched instances have frame-local negative IDs displayed as `seg N`.
The model receives both boxes and masks. Status text explains when fewer
than two objects are visible or no relations exceed the score threshold.

Relations also appear as `RA: person#3 holding cell phone#7 (score)` in the video
overlay and as `[relations]` console entries. `RELATION` records are written to
the existing SQLite events table and windowed JSON scene events. Their frame
index and track IDs refer to the analyzed frame, not the later display frame.
These are model predictions, not verified incidents. They do not automatically
trigger threat alerts or add messages to Bitchat; the existing VLM image and
description delivery stays in place.

One background job runs at a time, with no frame backlog. Scheduling is in
`configs/pipeline.yaml` under `router.stages.relations` (every 30 frames).
The model, vocabulary, minimum ranking score and box limit are in
`configs/models.yaml` under `relations`. To disable this feature, set
`features.relations: false`. An initialization failure is reported explicitly
and leaves the existing detector/VLM pipeline running without relation outputs.

The source is in `vendor/RelateAnything`, with upstream license and attribution
files retained (source revision `06766fdf56752ca535fc9b971fca99ce563676d0`).
Setup pins the downloaded model revision to
`2db90096be5217bdc7a9003c042950f45723d105`.
On a fresh setup, use the project Python environment:

```powershell
& '.\.venv\Scripts\python.exe' -m pip install --no-deps --no-build-isolation -e vendor/RelateAnything
& '.\.venv\Scripts\python.exe' tools/setup_relations.py
```

Dependencies are deliberately not upgraded: replacing Torch, Transformers or
OpenCV with upstream training pins can break the existing CUDA/VLM setup.
Model loading is local after setup. Upstream documentation and model licensing:
https://github.com/Maelic/RelateAnything
