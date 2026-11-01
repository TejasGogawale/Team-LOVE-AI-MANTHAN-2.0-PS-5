"""Camera failover tests without network access or camera hardware."""
import numpy as np
import pytest

from core.video_source import ESP32Source, WebcamSource, build_source


class Camera:
    def __init__(self, *args):
        self.closed = False
        self.frame = np.full((16, 16, 3), 1, dtype=np.uint8)

    def isOpened(self):
        return not self.closed

    def read(self):
        return True, self.frame

    def set(self, *args):
        return True

    def release(self):
        self.closed = True


def test_unconfigured_source_uses_webcam(monkeypatch):
    import core.video_source as sources
    monkeypatch.delenv("ESP32_CAM_URL", raising=False)
    monkeypatch.setattr(sources.cv2, "VideoCapture", Camera)
    source = build_source({"type": "esp32", "path": None})
    source.open()
    assert source.read()[0]
    source.release()
    assert not source.isOpened()


def test_existing_webcam_factory_unchanged():
    assert type(build_source({"type": "webcam"})) is WebcamSource


@pytest.mark.parametrize("initially_available", [True, False])
def test_fallback_and_recovery(monkeypatch, initially_available):
    import esp32_cam as esp
    import core.video_source as sources

    class Reader:
        instance = None

        def __init__(self, url):
            Reader.instance = self
            self.url = url
            self.available = initially_available
            self.frame = np.full((16, 16, 3), 2, dtype=np.uint8)
            self.stopped = False

        def start(self):
            pass

        def alive(self):
            return self.available

        def latest(self):
            return self.frame, 1, 0

        def wait_newer(self, *args, **kwargs):
            return self.frame, 2, 0

        def stop(self):
            self.stopped = True

    monkeypatch.setattr(esp, "_EspReader", Reader)
    monkeypatch.setattr(esp, "STARTUP_WAIT_S", 0)
    monkeypatch.setattr(esp, "FALLBACK_TO_WEBCAM", True)
    monkeypatch.setattr(esp, "RESIZE_MODE", "letterbox")
    monkeypatch.setenv("ESP32_CAM_URL", "http://camera.test:81/stream")
    monkeypatch.setattr(sources.cv2, "VideoCapture", Camera)
    source = ESP32Source(width=32, height=24)
    source.open()
    assert Reader.instance.url == "http://camera.test:81/stream"
    assert sources.cv2.VideoCapture is Camera
    for available in [initially_available, False, True, False]:
        Reader.instance.available = available
        ok, frame = source.read()
        assert ok
        assert frame.max() == (2 if available else 1)
        if available:
            assert frame.shape == (24, 32, 3)
    webcam = source.cap._webcam
    source.release()
    assert Reader.instance.stopped
    assert webcam.closed
    assert not source.isOpened()
