"""Tests for downloading and installing packs from inside the app.

No network, no 7-Zip, no real install: the HTTP opener and the extractor are
injected, and every root is a tmp dir. What's pinned down: bad bytes never
reach packs/ or models/, an interrupted download continues rather than
restarting, and a PyTorch pack never replaces one the running app has loaded.
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import urllib.error

import pytest

from modules.packs import pack_manager as pm


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


DATA = os.urandom(3 * 1024 * 1024 + 123)   # a few chunks, not a multiple of one


def _pack(name="torch-cu128", data=DATA, dep="torch", python=None):
    return pm.Pack(name=name, dep=dep, version="2.7.1+cu128", python=python,
                   asset=f"component-{name}.7z", bytes=len(data), sha256=_sha(data),
                   bytes_installed=10, url=f"https://host/packs/component-{name}.7z")


class _Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {}

    def getcode(self):
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


class Server:
    """Serves one body; honours Range unless told not to; can cut off or fail."""

    def __init__(self, data, honour_range=True, cut_after=None, fail=()):
        self.data = data
        self.honour_range = honour_range
        self.cut_after = cut_after       # bytes per response before "disconnect"
        self.fail = list(fail)           # exceptions to raise, in order
        self.calls = []

    def __call__(self, url, headers):
        self.calls.append(dict(headers))
        if self.fail:
            raise self.fail.pop(0)
        rng = headers.get("Range")
        if rng and self.honour_range:
            start = int(rng.split("=")[1].rstrip("-"))
            body = self.data[start:]
            resp = _Response(body[: self.cut_after] if self.cut_after else body, 206,
                             {"Content-Range": f"bytes {start}-{len(self.data) - 1}/{len(self.data)}"})
        else:
            body = self.data
            resp = _Response(body[: self.cut_after] if self.cut_after else body, 200)
        return resp


def _no_sleep(_):
    pass


@pytest.fixture(autouse=True)
def _isolated_roots(tmp_path, monkeypatch):
    """Keep every root the module or the loader looks at inside tmp."""
    exe = tmp_path / "app"
    user = tmp_path / "user"
    exe.mkdir()
    monkeypatch.setattr(pm, "exe_dir", lambda: str(exe))
    monkeypatch.setattr(pm, "per_user_dir", lambda: str(user))
    monkeypatch.setattr(pm, "target_root", lambda: str(exe))
    return exe, user


# --- lock -------------------------------------------------------------------

def test_load_lock_builds_urls_and_skips_bad_rows(tmp_path):
    lock = tmp_path / "packs.lock.json"
    lock.write_text(json.dumps({"base": "https://h/packs-x/", "packs": [
        {"name": "torch-cu128", "dep": "torch", "version": "v", "python": "cp312",
         "asset": "a.7z", "bytes": 5, "sha256": "AB" * 32, "bytes_installed": 7.0},
        {"name": "broken", "asset": "b.7z"},
    ]}), encoding="utf-8")
    packs = pm.load_lock(str(lock))
    assert list(packs) == ["torch-cu128"]
    p = packs["torch-cu128"]
    assert p.url == "https://h/packs-x/a.7z"
    assert p.sha256 == "ab" * 32 and p.bytes_installed == 7


def test_no_lock_means_nothing_offered(tmp_path):
    assert pm.load_lock(str(tmp_path / "missing.json")) == {}


# --- download ---------------------------------------------------------------

def test_download_verifies_and_lands(tmp_path):
    pack = _pack()
    path = pm.download_pack(pack, str(tmp_path), opener=Server(DATA), sleep=_no_sleep)
    assert open(path, "rb").read() == DATA
    assert not os.path.exists(path + ".part")


def test_download_resumes_with_range(tmp_path):
    pack = _pack()
    half = len(DATA) // 2
    (tmp_path / (pack.asset + ".part")).write_bytes(DATA[:half])
    server = Server(DATA)
    path = pm.download_pack(pack, str(tmp_path), opener=server, sleep=_no_sleep)
    assert server.calls[0]["Range"] == f"bytes={half}-"
    assert open(path, "rb").read() == DATA


def test_server_ignoring_range_restarts_cleanly(tmp_path):
    pack = _pack()
    (tmp_path / (pack.asset + ".part")).write_bytes(DATA[:1000])
    path = pm.download_pack(pack, str(tmp_path), opener=Server(DATA, honour_range=False),
                            sleep=_no_sleep)
    assert open(path, "rb").read() == DATA      # not 1000 bytes written twice


def test_dropped_connections_continue_where_they_stopped(tmp_path):
    pack = _pack()
    server = Server(DATA, cut_after=1024 * 1024)
    path = pm.download_pack(pack, str(tmp_path), opener=server, sleep=_no_sleep)
    assert open(path, "rb").read() == DATA
    assert len(server.calls) == 4               # 1 MB per response, then the tail
    assert all("Range" in c for c in server.calls[1:])


def test_transient_errors_are_retried(tmp_path):
    pack = _pack()
    server = Server(DATA, fail=[urllib.error.URLError("reset"), TimeoutError("slow")])
    path = pm.download_pack(pack, str(tmp_path), opener=server, sleep=_no_sleep)
    assert open(path, "rb").read() == DATA


def test_404_is_not_retried(tmp_path):
    pack = _pack()
    err = urllib.error.HTTPError(pack.url, 404, "Not Found", {}, None)
    server = Server(DATA, fail=[err])
    with pytest.raises(pm.PackError, match="404"):
        pm.download_pack(pack, str(tmp_path), opener=server, sleep=_no_sleep)
    assert len(server.calls) == 1


def test_hash_mismatch_is_discarded(tmp_path):
    pack = _pack()
    bad = bytes(len(DATA))
    dest = tmp_path / "dl"
    with pytest.raises(pm.PackError, match="checksum"):
        pm.download_pack(pack, str(dest), opener=Server(bad), sleep=_no_sleep)
    assert os.listdir(dest) == []


def test_cancel_keeps_the_partial_file(tmp_path):
    pack = _pack()
    seen = {"n": 0}

    def cancel():
        seen["n"] += 1
        return seen["n"] > 2        # after the first chunk or so

    assert pm.download_pack(pack, str(tmp_path), opener=Server(DATA),
                            should_cancel=cancel, sleep=_no_sleep) is None
    part = tmp_path / (pack.asset + ".part")
    assert 0 < part.stat().st_size < len(DATA)


# --- install ----------------------------------------------------------------

def _torch_extractor(priority):
    def extract(archive, dest):
        os.makedirs(os.path.join(dest, "site-packages", "torch"))
        with open(os.path.join(dest, "pack.json"), "w") as fh:
            json.dump({"name": "torch-cu128", "dep": "torch",
                       "version": "2.7.1+cu128", "priority": priority}, fh)
    return extract


def _make_installed(root, name, dep="torch", priority=0, version="2.7.1+cpu"):
    folder = root / "packs" / name
    (folder / "site-packages").mkdir(parents=True)
    (folder / "pack.json").write_text(json.dumps(
        {"name": name, "dep": dep, "priority": priority, "version": version}))
    return folder


def test_install_torch_pack_beside_cpu_and_mark_cpu(_isolated_roots):
    exe, _ = _isolated_roots
    cpu = _make_installed(exe, "torch-cpu")
    pack = _pack()
    result = pm.install_pack(pack.name, lock={pack.name: pack}, opener=Server(DATA),
                             extractor=_torch_extractor(10), sleep=_no_sleep)
    assert result.ok and result.restart_required
    assert (exe / "packs" / "torch-cu128" / "site-packages").is_dir()
    assert (cpu / ".remove").read_text() == "torch-cu128"
    assert not (exe / pm.STAGING_DIRNAME).exists()
    assert pm.status(pack) == pm.INSTALLED


def test_reinstall_goes_to_pending(_isolated_roots):
    exe, _ = _isolated_roots
    _make_installed(exe, "torch-cu128", priority=10, version="2.7.0+cu128")
    pack = _pack()
    assert pm.status(pack) == pm.OUTDATED
    result = pm.install_pack(pack.name, lock={pack.name: pack}, opener=Server(DATA),
                             extractor=_torch_extractor(10), sleep=_no_sleep)
    assert result.ok and result.restart_required
    assert (exe / "packs" / "torch-cu128.pending" / "pack.json").is_file()
    assert pm.status(pack) == pm.PENDING


def test_install_model_pack(_isolated_roots):
    exe, _ = _isolated_roots
    pack = _pack("models-clip", dep="models-clip")

    def extract(archive, dest):
        d = os.path.join(dest, "models", "clip-vit-base-patch32-ov")
        os.makedirs(d)
        open(os.path.join(d, "openvino_model.xml"), "w").close()

    result = pm.install_pack(pack.name, lock={pack.name: pack}, opener=Server(DATA),
                             extractor=extract, sleep=_no_sleep)
    assert result.ok and not result.restart_required
    assert (exe / "models" / "clip-vit-base-patch32-ov" / "openvino_model.xml").is_file()


def test_failed_unpack_keeps_verified_archive_for_retry(_isolated_roots):
    exe, _ = _isolated_roots
    pack = _pack()

    def broken(archive, dest):
        raise pm.PackError("7-Zip could not unpack")

    result = pm.install_pack(pack.name, lock={pack.name: pack}, opener=Server(DATA),
                             extractor=broken, sleep=_no_sleep)
    assert not result.ok and "7-Zip" in result.message
    assert (exe / pm.STAGING_DIRNAME / pack.asset).is_file()
    assert not (exe / "packs" / "torch-cu128").exists()

    server = Server(DATA)
    result = pm.install_pack(pack.name, lock={pack.name: pack}, opener=server,
                             extractor=_torch_extractor(10), sleep=_no_sleep)
    assert result.ok and server.calls == []      # nothing downloaded twice


def test_wrong_python_is_refused(_isolated_roots):
    pack = _pack(python="cp27")
    result = pm.install_pack(pack.name, lock={pack.name: pack}, opener=Server(DATA))
    assert not result.ok and "Python" in result.message


def test_not_enough_space(_isolated_roots, monkeypatch):
    pack = _pack()
    monkeypatch.setattr(pm, "_free_bytes", lambda path: 1024)
    result = pm.install_pack(pack.name, lock={pack.name: pack}, opener=Server(DATA))
    assert not result.ok and "space" in result.message


# --- NVIDIA -----------------------------------------------------------------

def _smi(stdout, rc=0):
    def run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, rc, stdout=stdout, stderr="")
    return run


def test_probe_nvidia_parses_names_with_commas(monkeypatch):
    monkeypatch.setattr(pm.shutil, "which", lambda n: "/usr/bin/nvidia-smi")
    gpus = pm.probe_nvidia(_smi("NVIDIA GeForce RTX 4070, 572.16\nQuadro, Inc X, 531.14\n"))
    assert gpus == [pm.NvidiaGpu("NVIDIA GeForce RTX 4070", "572.16"),
                    pm.NvidiaGpu("Quadro, Inc X", "531.14")]


def test_cuda_advice():
    assert "No NVIDIA" in pm.cuda_pack_advice([])
    assert "too old" in pm.cuda_pack_advice([pm.NvidiaGpu("GTX 1060", "472.12")])
    assert pm.cuda_pack_advice([pm.NvidiaGpu("RTX 5080", "576.02")]) is None


# --- the loader (packaging/packs/rthook_packs.py) ---------------------------

@pytest.fixture
def rthook(tmp_path, monkeypatch):
    """Import the runtime hook against tmp roots, without touching sys.path."""
    exe = tmp_path / "app"
    exe.mkdir(exist_ok=True)
    monkeypatch.setattr(sys, "executable", str(exe / "VideoHighlighter.exe"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "user"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "user"))
    monkeypatch.setattr(sys, "path", list(sys.path))
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "rthook_packs_under_test", os.path.join(root, "packaging", "packs", "rthook_packs.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, exe


def test_loader_prefers_gpu_and_ignores_pending(rthook):
    hook, exe = rthook
    _make_installed(exe, "torch-cpu", priority=0)
    _make_installed(exe, "torch-cu128", priority=10)
    pending = exe / "packs" / "torch-cu128.pending"
    (pending / "site-packages").mkdir(parents=True)
    (pending / "pack.json").write_text(json.dumps({"dep": "torch", "priority": 99}))
    chosen = hook._chosen_packs([str(exe / "packs")])
    assert [name for _, name, _ in chosen] == ["torch-cu128"]


def test_loader_swaps_pending_and_clears_superseded(rthook):
    hook, exe = rthook
    cpu = _make_installed(exe, "torch-cpu", priority=0)
    (cpu / ".remove").write_text("torch-cu128")
    _make_installed(exe, "torch-cu128", priority=10, version="old")
    pending = _make_installed(exe, "torch-cu128.pending", priority=10, version="new")
    hook._maintain([str(exe / "packs")])
    packs = exe / "packs"
    assert sorted(os.listdir(packs)) == ["torch-cu128"]
    assert json.loads((packs / "torch-cu128" / "pack.json").read_text())["version"] == "new"
    assert not pending.exists()


def test_loader_keeps_cpu_when_replacement_is_missing(rthook):
    hook, exe = rthook
    cpu = _make_installed(exe, "torch-cpu", priority=0)
    (cpu / ".remove").write_text("torch-cu128")
    hook._maintain([str(exe / "packs")])
    assert cpu.is_dir()


def test_workers_do_no_maintenance(rthook, monkeypatch):
    hook, _ = rthook
    monkeypatch.setattr(sys, "argv", ["VideoHighlighter.exe", "--multiprocessing-fork",
                                      "parent_pid=1", "pipe_handle=2"])
    assert hook._is_worker()
