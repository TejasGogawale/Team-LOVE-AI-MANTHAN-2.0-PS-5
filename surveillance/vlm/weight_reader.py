"""Tensor-at-a-time fallback for Windows checkpoint commit-limit failures."""
import json
import math
import struct
from contextlib import contextmanager
from unittest.mock import patch


class StreamedWeights:
    def __init__(self, filename):
        self.filename = filename
        with open(filename, "rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            if size > 100_000_000:
                raise ValueError("Invalid safetensors header size")
            self.header = json.loads(stream.read(size))
        self.offset = 8 + size

    def keys(self):
        return [key for key in self.header if key != "__metadata__"]

    def metadata(self):
        return self.header.get("__metadata__")

    def get_slice(self, key):
        return WeightSlice(self, self.header[key])

    def get_tensor(self, key):
        return self.get_slice(key)[...]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class WeightSlice:
    def __init__(self, reader, entry):
        self.reader, self.entry = reader, entry

    def get_shape(self):
        return self.entry["shape"]

    def get_dtype(self):
        return self.entry["dtype"]

    def __getitem__(self, index):
        import torch
        dtypes = {"BF16": torch.bfloat16, "F16": torch.float16,
                  "F32": torch.float32, "F64": torch.float64,
                  "I64": torch.int64, "I32": torch.int32,
                  "I16": torch.int16, "I8": torch.int8,
                  "U8": torch.uint8, "BOOL": torch.bool}
        dtype = dtypes[self.get_dtype()]
        start, end = self.entry["data_offsets"]
        expected = math.prod(self.get_shape()) * torch.empty((), dtype=dtype).element_size()
        if start < 0 or end - start != expected:
            raise ValueError("Invalid safetensors tensor offsets")
        if expected == 0:
            return torch.empty(self.get_shape(), dtype=dtype)[index]
        with open(self.reader.filename, "rb") as stream:
            stream.seek(self.reader.offset + start)
            data = bytearray(stream.read(expected))
        if len(data) != expected:
            raise ValueError("Truncated safetensors tensor")
        return torch.frombuffer(data, dtype=dtype).reshape(self.get_shape())[index]


@contextmanager
def windows_checkpoint_fallback():
    import sys
    from transformers import modeling_utils
    original = modeling_utils.safe_open

    def open_weights(filename, *args, **kwargs):
        try:
            return original(filename, *args, **kwargs)
        except OSError as exc:
            if sys.platform != "win32" or not (
                getattr(exc, "winerror", None) == 1455 or "1455" in str(exc)
            ):
                raise
            print("[VLM] Windows checkpoint open hit commit limit; reading weights one tensor at a time.")
            return StreamedWeights(filename)

    with patch.object(modeling_utils, "safe_open", open_weights):
        yield
