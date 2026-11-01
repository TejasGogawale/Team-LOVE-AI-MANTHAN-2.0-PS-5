"""Scene delivery regressions; no model weights or phone required."""
from unittest.mock import Mock

import numpy as np
import pytest


def test_description_is_delivered_before_entity_generation_fails():
    from vlm.core import VLMCore, VLMConfig

    core = VLMCore(VLMConfig())
    core._model = object()
    core._processor = object()
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    delivered = []
    core.on_scene_ready = lambda text, image: delivered.append((text, image))

    def prompt(image, text):
        if text == core.SCENE_PROMPT:
            return "A person at a desk."
        assert delivered[0][0] == "A person at a desk."
        assert delivered[0][1] is frame
        raise RuntimeError("entity generation failed")

    core._run_single_prompt = prompt
    with pytest.raises(RuntimeError, match="entity generation failed"):
        core._run_model_inference(frame)
    assert len(delivered) == 1


def test_scene_image_and_text_both_respect_rate_limit():
    from core.bitchat import BitchatAlertClient

    client = BitchatAlertClient.__new__(BitchatAlertClient)
    client._enqueue = Mock()
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    client.send_scene("A person.", frame)
    calls = client._enqueue.call_args_list
    assert [call.args[0] for call in calls] == ["/send/image", "/send/text"]
    assert all(not call.kwargs.get("skip_rate", False) for call in calls)
    assert calls[0].args[2] is frame


@pytest.mark.parametrize("endpoint", ["/send/image", "/send/text"])
@pytest.mark.parametrize("status_code,body", [(500, {}), (200, {"status": "error"})])
def test_rejected_send_raises(monkeypatch, endpoint, status_code, body):
    from core.bitchat import BitchatAlertClient
    import core.bitchat as bitchat

    client = BitchatAlertClient.__new__(BitchatAlertClient)
    client.base_url = "http://test.invalid"
    client.timeout = 1
    client.channel = None
    response = Mock(status_code=status_code, text="rejected")
    response.json.return_value = body
    monkeypatch.setattr(bitchat.requests, "post", Mock(return_value=response))
    frame = np.zeros((16, 16, 3), dtype=np.uint8) if endpoint.endswith("image") else None
    with pytest.raises(RuntimeError):
        client._send(endpoint, {"text": "A person."}, frame)
