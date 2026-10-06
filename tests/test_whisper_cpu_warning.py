"""A large Whisper model on the processor is called out before it starts.

On a CPU it can take longer than the video, with a bar that moves once per 30 s
of decoded audio; people took that for a hang and closed the app mid-chunk.
"""
from __future__ import annotations

import pytest

from modules.audio.transcript import cpu_model_warning


@pytest.mark.parametrize("model", ["medium", "large", "large-v3"])
def test_slow_models_on_the_processor_are_named(model):
    line = cpu_model_warning(model, "cpu")
    assert line and model in line and "processor" in line


@pytest.mark.parametrize("model,device", [
    ("large", "cuda"), ("small", "cpu"), ("base", "cpu"), ("tiny", "cpu"),
])
def test_quiet_otherwise(model, device):
    assert cpu_model_warning(model, device) is None
