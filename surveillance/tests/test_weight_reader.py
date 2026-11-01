import pytest
import torch
from safetensors.torch import save_file
from vlm.weight_reader import StreamedWeights, windows_checkpoint_fallback


def test_streamed_weights_match_original_tensors(tmp_path):
    path = tmp_path / "weights.safetensors"
    tensors = {"weight": torch.arange(24, dtype=torch.bfloat16).reshape(4, 6),
               "bias": torch.arange(6, dtype=torch.float32),
               "ids": torch.tensor([1, 3], dtype=torch.int64)}
    save_file(tensors, str(path), metadata={"format": "pt"})
    reader = StreamedWeights(path)
    assert set(reader.keys()) == set(tensors)
    assert reader.metadata() == {"format": "pt"}
    for key, original in tensors.items():
        assert torch.equal(reader.get_tensor(key), original)
    assert torch.equal(reader.get_slice("weight")[1:3], tensors["weight"][1:3])


def test_windows_commit_error_falls_back_and_restores(monkeypatch, tmp_path):
    import sys
    from transformers import modeling_utils
    path = tmp_path / "weights.safetensors"
    save_file({"x": torch.ones(2)}, str(path))
    def failing(*args, **kwargs):
        raise OSError("paging file too small (os error 1455)")
    monkeypatch.setattr(modeling_utils, "safe_open", failing)
    monkeypatch.setattr(sys, "platform", "win32")
    with windows_checkpoint_fallback():
        reader = modeling_utils.safe_open(path, framework="pt")
        assert torch.equal(reader.get_tensor("x"), torch.ones(2))
    assert modeling_utils.safe_open is failing


def test_other_errors_are_not_hidden(monkeypatch):
    from transformers import modeling_utils
    def failing(*args, **kwargs):
        raise OSError("checkpoint missing")
    monkeypatch.setattr(modeling_utils, "safe_open", failing)
    with windows_checkpoint_fallback(), pytest.raises(OSError, match="checkpoint missing"):
        modeling_utils.safe_open("missing")
