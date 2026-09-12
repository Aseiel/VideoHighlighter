"""
Tests for modules.cuda_check — CUDA that is "available" but cannot run.

Observed on a user's RTX 5060 under the release build's CUDA 12.4 torch:
is_available() said yes, the build's kernels stop at sm_90, and the first real
operation failed with "no kernel image is available for execution on the
device". Fakes stand in for torch throughout; nothing here needs a GPU.
"""

from __future__ import annotations

import types

from modules import cuda_check

CU124_ARCHS = ["sm_50", "sm_60", "sm_61", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]


def fake_torch(capability=(8, 9), archs=CU124_ARCHS, available=True,
               name="NVIDIA GeForce RTX 4070", smoke_error=None):
    ran = []

    def zeros(*_args, **kwargs):
        if smoke_error:
            raise RuntimeError(smoke_error)
        ran.append(kwargs.get("device"))
        return types.SimpleNamespace(add_=lambda value: None)

    cuda = types.SimpleNamespace(
        is_available=lambda: available,
        device_count=lambda: 1 if available else 0,
        get_arch_list=lambda: list(archs),
        get_device_capability=lambda index=0: capability,
        get_device_name=lambda index=0: name,
        get_device_properties=lambda index=0: types.SimpleNamespace(total_mem=8 * 1024 ** 3),
        synchronize=lambda: None,
    )
    return types.SimpleNamespace(cuda=cuda, zeros=zeros, ran=ran)


def test_a_card_newer_than_the_build_is_not_usable():
    torch = fake_torch(capability=(12, 0), name="NVIDIA GeForce RTX 5060")

    reason = cuda_check.cuda_unusable_reason(torch)

    assert reason and "RTX 5060" in reason and "sm_120" in reason
    assert not cuda_check.cuda_usable(torch)
    assert torch.ran == []  # decided without touching the card


def test_a_supported_card_is_usable_and_proven_by_running_something():
    torch = fake_torch(capability=(8, 9))  # sm_89, covered by the sm_8x kernels

    assert cuda_check.cuda_usable(torch)
    assert torch.ran == ["cuda"]


def test_a_failing_test_operation_is_not_usable():
    torch = fake_torch(smoke_error="CUDA error: no kernel image is available "
                                   "for execution on the device")

    assert "no kernel image" in cuda_check.cuda_unusable_reason(torch)


def test_no_device_is_not_usable():
    assert not cuda_check.cuda_usable(fake_torch(available=False))


def test_a_torch_that_cannot_describe_itself_gets_the_benefit_of_the_doubt():
    bare = types.SimpleNamespace(cuda=types.SimpleNamespace(
        is_available=lambda: True, device_count=lambda: 1))

    assert cuda_check.cuda_usable(bare)


def test_the_answer_is_cached_per_torch():
    torch = fake_torch()

    cuda_check.cuda_usable(torch)
    cuda_check.cuda_usable(torch)

    assert torch.ran == ["cuda"]


def test_detect_best_device_passes_over_an_unusable_card(monkeypatch):
    from modules import device_utils as du
    torch = fake_torch(capability=(12, 0), name="NVIDIA GeForce RTX 5060")
    monkeypatch.setattr(du, "_TORCH_AVAILABLE", True)
    monkeypatch.setattr(du, "torch", torch)

    said = []
    info = du.detect_best_device(log_fn=said.append)

    assert info.backend_name != "CUDA"
    assert not info.yolo_pt_device.startswith("cuda")
    assert not info.motion_device.startswith("cuda")
    assert any("RTX 5060" in line for line in said)


def test_resolve_yolo_device_refuses_an_unusable_card(monkeypatch):
    from modules import device_utils as du
    monkeypatch.setattr(du, "_TORCH_AVAILABLE", True)
    monkeypatch.setattr(du, "torch", fake_torch(capability=(12, 0)))

    assert du.resolve_yolo_device("cuda:0") == "cpu"
