"""Load Qwen without optional training losses when Windows blocks SciPy."""
from __future__ import annotations

from unittest.mock import patch


def load_qwen_classes():
    import transformers.utils as utils

    if utils.is_scipy_available():
        try:
            from scipy.optimize import linear_sum_assignment  # noqa: F401
        except ImportError as exc:
            # Do not hide unrelated dependency failures or change Windows policy.
            if "Application Control policy has blocked" not in str(exc):
                raise
            print(
                "[VLM] Windows blocked an optional SciPy DLL; loading Qwen "
                "without SciPy-dependent object detection training losses."
            )
            # Transformers checks this exported predicate before importing its
            # optional losses. Restore it immediately after loading Qwen; other
            # pipeline components must retain their normal SciPy behavior.
            with patch.object(utils, "is_scipy_available", return_value=False):
                from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
                return Qwen2_5_VLForConditionalGeneration, AutoProcessor

    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    return Qwen2_5_VLForConditionalGeneration, AutoProcessor
