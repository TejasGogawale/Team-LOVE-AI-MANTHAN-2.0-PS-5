from types import SimpleNamespace
from threading import Event as ThreadEvent
import time

import numpy as np

from core.relations import RelationPredictor, BackgroundRelations


def track(tid, cls, box, conf=0.9):
    return SimpleNamespace(track_id=tid, cls=cls, xyxy=box, conf=conf)


def test_box_mapping_scores_and_event_details():
    class Model:
        def predict(self, frame, boxes, **kwargs):
            np.testing.assert_equal(boxes, [[0, 0, 30, 40], [35, 5, 60, 40]])
            assert kwargs["box_labels"] == ["person", "phone"]
            assert kwargs["max_boxes"] == 2
            assert "box_scores" not in kwargs
            return [SimpleNamespace(subject_idx=0, object_idx=1, predicate="holding", score=0.8),
                    SimpleNamespace(subject_idx=1, object_idx=0, predicate="near", score=0.1)]
    predictor = RelationPredictor({"max_boxes": 2, "min_score": 0.25}, Model())
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    events = predictor.predict(image, [track(7, 1, (35, 5, 70, 45), 0.8),
                                      track(3, 0, (-5, -1, 30, 40)),
                                      track(9, 0, (0, 0, 10, 10), 0.2),
                                      track(4, 0, (20, 20, 10, 10))], 42, ["person", "phone"])
    assert len(events) == 1
    assert events[0].frame_idx == 42
    assert events[0].details["subject_track_id"] == 3
    assert events[0].details["object_track_id"] == 7
    assert events[0].details["description"] == "person#3 holding phone#7"


def test_empty_or_invalid_regions_skip_model():
    class Model:
        def predict(self, *args, **kwargs):
            raise AssertionError("Model should not run")
    predictor = RelationPredictor({}, Model())
    assert predictor.predict(np.zeros((40, 60, 3)),
        [track(1, 0, (0, 0, 10, 10)), track(2, 0, (float('nan'), 0, 3, 3))], 0, ["person"]) == []
    assert predictor.status == "waiting for 2 objects (found 1)"


def test_background_is_bounded_and_copies_input():
    gate = ThreadEvent()
    class Predictor:
        def predict(self, image, tracks, frame_idx, labels):
            assert gate.wait(2)
            assert image[0, 0, 0] == 0
            assert tracks[0].track_id == 1
            return [frame_idx]
    worker = BackgroundRelations(Predictor(), ["person"])
    try:
        image = np.zeros((2, 2, 3))
        tracks = [track(1, 0, (0, 0, 1, 1))]
        assert worker.submit(image, tracks, 42)
        image[:] = 1
        tracks[0].track_id = 9
        assert not worker.submit(image, tracks, 43)
        assert worker.poll() is None
        gate.set()
        assert worker.future.result(timeout=2) == [42]
        assert worker.poll() == [42]
        assert worker.poll() is None
    finally:
        gate.set()
        worker.close()


def test_relation_events_survive_json_flush(tmp_path):
    from core.event_buffer import EventBuffer
    from core.events import Event
    buffer = EventBuffer({"json_dir": str(tmp_path)})
    event = Event("RELATION", time.strftime("%Y-%m-%dT%H:%M:%S"), 42,
                  {"predicate": "holding", "subject_track_id": 1, "object_track_id": 2, "score": 0.7})
    buffer.append(event, frame_idx=99)
    saved = buffer.force_flush(99)
    assert saved["scene_events"][0]["frame_idx"] == 42
    assert saved["scene_events"][0]["details"]["object_track_id"] == 2
