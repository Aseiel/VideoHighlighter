"""The web build's sidecar fetches NVIDIA PyTorch and CLIP itself, unasked."""
from __future__ import annotations

from types import SimpleNamespace

from modules.packs import pack_manager as pm
from sidecar import packs_fetch


def _lock(*names):
    return {n: SimpleNamespace(download_mb=10) for n in names}


def _ready(*names):
    return lambda name, _lock: name in names


def _gpu():
    return [pm.NvidiaGpu("A card", "580.10")]


def test_fetches_clip_and_cuda_on_a_usable_nvidia_card():
    wanted = packs_fetch.wanted_packs(
        _lock(pm.CPU_PACK, pm.CUDA_PACK, pm.CLIP_PACK),
        is_ready=_ready(pm.CPU_PACK), probe=_gpu, advice=pm.cuda_pack_advice,
        log=lambda _m: None)
    assert wanted == [pm.CLIP_PACK, pm.CUDA_PACK]


def test_no_nvidia_card_means_no_2gb_download_and_says_why():
    said = []
    wanted = packs_fetch.wanted_packs(
        _lock(pm.CUDA_PACK, pm.CLIP_PACK),
        is_ready=_ready(), probe=lambda: [], advice=pm.cuda_pack_advice,
        log=said.append)
    assert wanted == [pm.CLIP_PACK]
    assert any(pm.CUDA_PACK in m and "not fetched" in m for m in said)


def test_an_old_driver_is_not_offered_the_pack():
    wanted = packs_fetch.wanted_packs(
        _lock(pm.CUDA_PACK), is_ready=_ready(),
        probe=lambda: [pm.NvidiaGpu("Old", "400.00")], advice=pm.cuda_pack_advice,
        log=lambda _m: None)
    assert wanted == []


def test_nothing_installed_twice():
    wanted = packs_fetch.wanted_packs(
        _lock(pm.CUDA_PACK, pm.CLIP_PACK),
        is_ready=_ready(pm.CUDA_PACK, pm.CLIP_PACK), probe=_gpu,
        advice=pm.cuda_pack_advice, log=lambda _m: None)
    assert wanted == []


def test_fetch_installs_in_order_and_reports_each():
    done, said = [], []

    def install(name, *, lock, progress):
        progress(pm.DOWNLOADING, 5, 10, "asset")
        progress(pm.DOWNLOADING, 10, 10, "asset")
        done.append(name)
        return pm.PackResult(ok=name == pm.CLIP_PACK, message="no space", name=name,
                             restart_required=False)

    results = packs_fetch.fetch_missing(
        log=said.append, install=install,
        lock=_lock(pm.CUDA_PACK, pm.CLIP_PACK), is_ready=_ready(),
        probe=_gpu, advice=pm.cuda_pack_advice)
    assert done == [pm.CLIP_PACK, pm.CUDA_PACK]
    assert [r.ok for r in results] == [True, False]
    assert any("50%" in m for m in said)
    assert any(pm.CUDA_PACK in m and "no space" in m for m in said)


def test_a_build_without_a_lock_does_nothing():
    assert packs_fetch.fetch_missing(lock={}, install=None) == []


def test_turned_off_by_the_environment(monkeypatch):
    monkeypatch.setenv(packs_fetch.DISABLE_ENV, "1")
    assert packs_fetch.start_in_background(log=lambda _m: None) is None
