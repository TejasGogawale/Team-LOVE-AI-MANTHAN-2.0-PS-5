"""RelateAnything relations from the existing YOLO/ByteTrack regions."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import math
import time

import numpy as np
import cv2

from core.events import Event


@dataclass(frozen=True)
class Region:
    track_id: int
    xyxy: tuple
    conf: float
    label: str


class RelationPredictor:
    def __init__(self, config, model=None):
        self.config = config
        self.model = model
        self.status = "not loaded"
        self.segmenter = None
        self.visual_frame = None

    def initialize(self):
        if self.model is not None:
            self.status = "ready"
            return
        checkpoint = Path(self.config["checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(f"RelateAnything checkpoint missing: {checkpoint}")
        # Reuse the Windows optional-SciPy compatibility preparation. Importing
        # Qwen classes here allocates no weights and avoids blocked loss DLLs.
        from vlm.transformers_compat import load_qwen_classes
        load_qwen_classes()
        from relsgg import RelateAnything
        print(f"[relations] Loading RelateAnything on {self.config.get('device', 'cuda:0')}...", flush=True)
        self.model = RelateAnything.from_checkpoint(
            str(checkpoint), device=self.config.get("device", "cuda:0"),
            predicates=self.config.get("predicates"),
        )
        self.status = "ready"
        if self.config.get("segmentation", False):
            from ultralytics import YOLO
            self.segmenter = YOLO(self.config["segmentation_weights"])
            print("[relations] YOLO segmentation ready", flush=True)

    def _segment(self, frame, tracks):
        from types import SimpleNamespace
        result = self.segmenter.predict(frame, imgsz=480, conf=0.25,
            device=self.config.get("device", "cuda:0"),
            retina_masks=True, verbose=False)[0]
        if result.masks is None:
            return [], []
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        scores = result.boxes.conf.cpu().numpy()
        masks = result.masks.data.cpu().numpy() > 0.5
        regions = []
        used = set()
        for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
            best, best_iou = None, 0.3
            for ti, track in enumerate(tracks):
                if ti in used or int(track.cls) != cls:
                    continue
                other = np.asarray(track.xyxy)
                low, high = np.maximum(box[:2], other[:2]), np.minimum(box[2:], other[2:])
                overlap = np.maximum(high - low, 0).prod()
                area = (box[2:] - box[:2]).prod() + (other[2:] - other[:2]).prod() - overlap
                iou = overlap / max(area, 1)
                if iou > best_iou:
                    best, best_iou = ti, iou
            if best is not None:
                used.add(best)
            regions.append(SimpleNamespace(track_id=tracks[best].track_id if best is not None else -(i + 1),
                xyxy=tuple(box), cls=cls, conf=float(score), mask_index=i))
        return regions, masks

    def predict(self, frame, tracks, frame_idx, labels):
        masks = None
        if self.segmenter is not None:
            tracks, masks = self._segment(frame, tracks)
        height, width = frame.shape[:2]
        regions = []
        region_masks = {}
        for track in tracks:
            coords = np.asarray(track.xyxy, dtype=np.float32)
            if not np.isfinite(coords).all() or not math.isfinite(float(track.conf)):
                continue
            x1, y1, x2, y2 = coords
            box = (max(0, min(width, x1)), max(0, min(height, y1)),
                   max(0, min(width, x2)), max(0, min(height, y2)))
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            cls = int(track.cls)
            label = labels[cls] if 0 <= cls < len(labels) else str(cls)
            regions.append(Region(int(track.track_id), box, float(track.conf), label))
            if masks is not None:
                region_masks[int(track.track_id)] = masks[track.mask_index]
        regions.sort(key=lambda region: region.conf, reverse=True)
        regions = regions[:max(2, int(self.config.get("max_boxes", 12)))]
        if len(regions) < 2:
            self.status = f"waiting for 2 objects (found {len(regions)})"
            self.visual_frame = self.draw_visual(frame, regions, region_masks, [], frame_idx)
            return []
        self.status = f"analyzing {len(regions)} objects"
        # Upstream accepts OpenCV BGR arrays directly. Keep original pixel
        # coordinates; the API normalizes them internally.
        triplets = self.model.predict(
            frame, np.array([r.xyxy for r in regions], dtype=np.float32),
            box_labels=[r.label for r in regions],
            masks=[region_masks[r.track_id] for r in regions] if masks is not None else None,
            # Regions are already selected by detection confidence. Threshold
            # the relation score itself: multiplying by two detector scores
            # can suppress a strong relation between lower-confidence objects.
            max_boxes=len(regions), topk=int(self.config.get("topk", 5)),
        )
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        events = []
        for triplet in triplets:
            score = float(triplet.score)
            if not math.isfinite(score) or score < float(self.config.get("min_score", 0.25)):
                continue
            subject, obj = regions[int(triplet.subject_idx)], regions[int(triplet.object_idx)]
            description = (f"{subject.label}#{subject.track_id} "
                           f"{triplet.predicate} {obj.label}#{obj.track_id}")
            events.append(Event("RELATION", timestamp, frame_idx, {
                "subject_track_id": subject.track_id, "object_track_id": obj.track_id,
                "subject_label": subject.label, "object_label": obj.label,
                "subject_box": [float(v) for v in subject.xyxy],
                "object_box": [float(v) for v in obj.xyxy],
                "predicate": str(triplet.predicate), "score": score,
                "description": description, "source": "RelateAnything",
            }))
        self.status = f"{len(events)} relations from {len(regions)} objects"
        self.visual_frame = self.draw_visual(frame, regions, region_masks, events, frame_idx)
        return events

    @staticmethod
    def draw_visual(frame, regions, masks, events, frame_idx):
        vis = np.clip(frame, 0, 255).astype(np.uint8)
        for ri, region in enumerate(regions):
            color = ((ri * 67 + 80) % 256, (ri * 97 + 160) % 256, (ri * 43 + 230) % 256)
            mask = masks.get(region.track_id)
            if mask is not None:
                vis[mask] = (vis[mask] * 0.6 + np.array(color) * 0.4).astype(np.uint8)
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, contours, -1, color, 2)
            x1, y1, x2, y2 = [int(v) for v in region.xyxy]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
            ident = f"#{region.track_id}" if region.track_id >= 0 else f" [seg {-region.track_id}]"
            cv2.putText(vis, region.label + ident, (x1, max(38, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        for ei, event in enumerate(events):
            details = event.details
            subject, obj = details["subject_box"], details["object_box"]
            a = (int((subject[0] + subject[2])/2), int((subject[1] + subject[3])/2))
            b = (int((obj[0] + obj[2])/2), int((obj[1] + obj[3])/2))
            cv2.arrowedLine(vis, a, b, (0, 255, 255), 2, tipLength=0.08)
            point = ((a[0] + b[0])//2, max(45, min(vis.shape[0]-10,
                (a[1] + b[1])//2 + (ei - len(events)//2) * 22)))
            text = f"{details['predicate']} {details['score']:.2f}"
            cv2.putText(vis, text, point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
            cv2.putText(vis, text, point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(vis, f"RelateAnything | analyzed frame {frame_idx} | {len(events)} relations",
                    (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        return vis


class BackgroundRelations:
    """One in-flight frame, no backlog of stale video or blocking capture."""
    def __init__(self, predictor, labels):
        self.predictor, self.labels = predictor, labels
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="relations")
        self.future = None

    def submit(self, frame, tracks, frame_idx):
        if self.future is not None:
            return False
        from copy import copy
        self.future = self.executor.submit(self.predictor.predict, frame.copy(),
                                           [copy(t) for t in tracks], frame_idx, self.labels)
        return True

    def poll(self):
        if self.future is None or not self.future.done():
            return None
        future, self.future = self.future, None
        return future.result()

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
