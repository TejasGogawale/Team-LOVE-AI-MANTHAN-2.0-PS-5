"""Offline recovery and GPU pacing regressions without a phone or model weights."""
from unittest.mock import Mock

import numpy as np
import pytest
import requests


@pytest.fixture
def sender(monkeypatch):
    import core.bitchat as bc
    monkeypatch.setattr(bc.threading, "Thread", Mock())
    client = bc.BitchatAlertClient("test.invalid", max_queue=4)
    monkeypatch.setattr(client._stop, "wait", lambda seconds: False)
    return client


def test_offline_retry_preserves_image_before_description(sender):
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    sender.send_scene("A person.", frame)
    attempts = []

    def send(endpoint, data, image):
        attempts.append(endpoint)
        if len(attempts) == 1:
            raise requests.ConnectTimeout("phone offline")
        if len(attempts) == 3:
            sender._stop.set()

    sender._send = send
    sender._worker()
    assert attempts == ["/send/image", "/send/image", "/send/text"]


def test_offline_backlog_keeps_latest_scene_pairs(sender):
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    for i in range(10):
        sender.send_scene(str(i), frame)
    assert len(sender._queue) == 4
    assert [item[0] for item in sender._queue] == ["/send/image", "/send/text"] * 2
    assert [item[1].get("caption", item[1].get("text")) for item in sender._queue] == ["8", "8", "9", "9"]
    assert sender._queue[0][2] is not frame


def test_connection_timeout_propagates_without_traceback(sender, monkeypatch, capsys):
    import core.bitchat as bc
    monkeypatch.setattr(bc.requests, "post", Mock(side_effect=requests.ConnectTimeout("offline")))
    with pytest.raises(requests.ConnectTimeout):
        sender._send("/send/image", {"caption": "A person."}, np.zeros((16, 16, 3), dtype=np.uint8))
    assert "Traceback" not in capsys.readouterr().err


def test_noisy_mesh_image_stays_within_budget():
    from core.bitchat import _prepare_image
    frame = np.random.default_rng(0).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    assert len(_prepare_image(frame)) <= 24576


def test_stale_alert_is_discarded_before_sending(sender):
    sender.send_scene("stale")
    item = sender._queue.popleft()
    sender._queue.append((*item[:-1], item[-1] - sender.max_age_s - 1))
    sender.send_scene("current")
    sent = []

    def send(endpoint, data, image):
        sent.append(data["text"])
        sender._stop.set()

    sender._send = send
    sender._worker()
    assert sent == ["current"]


def test_priority_alert_still_waits_for_phone_rate_window(sender, monkeypatch):
    import core.bitchat as bc
    monkeypatch.setattr(bc.time, "perf_counter", lambda: 100.0)
    sender._last_sent_t = 100.0
    waits = []
    monkeypatch.setattr(sender._stop, "wait", lambda seconds: waits.append(seconds) or False)
    sender.send_alert("FALL", "person on floor", priority=True)
    sender._send = lambda *args: sender._stop.set()
    sender._worker()
    assert waits == [sender.rate_limit_s]


def test_close_releases_frames_and_rejects_new_messages(sender):
    sender.send_scene("pending", np.zeros((16, 16, 3), dtype=np.uint8))
    sender.close()
    sender.send_scene("after close")
    assert not sender._queue


@pytest.mark.parametrize("fail", [False, True])
def test_vlm_cooldown_starts_after_completion_even_on_error(monkeypatch, fail):
    import vlm.core as vlm
    now = [100.0]
    monkeypatch.setattr(vlm.time, "perf_counter", lambda: now[0])
    manager = vlm.VLMManager(vlm.VLMConfig(min_interval_s=10.0))

    def infer(*args):
        now[0] = 200.0  # simulate a long model generation
        if fail:
            raise RuntimeError("model failure")
        return vlm.VLMOutput(1, now[0], True, 1.0)

    manager._handle_escalation = infer
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    if fail:
        with pytest.raises(RuntimeError):
            manager.process_frame(frame, 1, 1.0)
    else:
        manager.process_frame(frame, 1, 1.0)
    assert manager.should_process(2, 1.0) == (False, "cooldown")
    now[0] = 210.0
    assert manager.should_process(3, 1.0)[0]


def test_periodic_vlm_uses_elapsed_time_instead_of_frame_zero(monkeypatch):
    import vlm.core as vlm
    now = [100.0]
    monkeypatch.setattr(vlm.time, "perf_counter", lambda: now[0])
    manager = vlm.VLMManager(vlm.VLMConfig(forced_interval_s=30.0))
    assert not manager.should_process(0)[0]
    now[0] = 130.0
    assert manager.should_process(17) == (True, "forced_interval")


def test_offline_status_is_concise_and_logged_only_on_change(sender, monkeypatch, capsys):
    import core.bitchat as bc
    monkeypatch.setattr(bc.requests, "get", Mock(side_effect=requests.ConnectTimeout("long urllib3 error")))
    assert not sender.check_connection()
    assert not sender.check_connection()
    log = capsys.readouterr().out
    assert log.count("offline:") == 1
    assert "ConnectTimeout" in log
    assert "long urllib3 error" not in log


@pytest.mark.parametrize("body", [
    {"status": "ok", "api_enabled": False, "mesh_running": True},
    {"status": "ok", "api_enabled": True, "mesh_running": False},
    {"api_enabled": True},
    [],
])
def test_status_requires_ready_api_and_mesh(sender, monkeypatch, body):
    import core.bitchat as bc
    response = Mock()
    response.json.return_value = body
    monkeypatch.setattr(bc.requests, "get", Mock(return_value=response))
    assert not sender.check_connection()


def test_worker_rechecks_offline_phone_and_resumes_queued_message(sender, monkeypatch):
    import core.bitchat as bc
    now = [100.0]
    monkeypatch.setattr(bc.time, "perf_counter", lambda: now[0])
    sender.reconnect_interval_s = 10.0
    sender._next_connection_check = 100.0
    offline = Mock()
    offline.json.return_value = {"status": "ok", "api_enabled": False}
    ready = Mock()
    ready.json.return_value = {"status": "ok", "api_enabled": True, "mesh_running": True}
    get = Mock(side_effect=[offline, ready])
    monkeypatch.setattr(bc.requests, "get", get)
    monkeypatch.setattr(sender._stop, "wait", lambda seconds: now.__setitem__(0, now[0] + seconds) or False)
    sender.send_scene("Waiting for phone")
    sent = []

    def send(endpoint, data, frame):
        assert now[0] >= 110.0
        sent.append(data["text"])
        sender._stop.set()

    sender._send = send
    sender._worker()
    assert get.call_count == 2
    assert sent == ["Waiting for phone"]


def test_worker_checks_connection_without_a_vlm_message(sender, monkeypatch):
    sender.reconnect_interval_s = 10.0
    sender._next_connection_check = 0.0
    def check():
        sender._connected = False
        sender._stop.set()
    sender.check_connection = Mock(side_effect=check)
    sender._worker()
    sender.check_connection.assert_called_once()
