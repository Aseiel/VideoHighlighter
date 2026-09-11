"""The ONNX Runtime DirectML probe: honest about which build is installed.

Three outcomes are all "no GPU" and all need different words. The package is
missing; the package is there but is the plain CPU build; the user turned
DirectML off. A message that collapses them sends an AMD user to install
something they already have.

No ONNX Runtime is needed to test any of this — a fake module in `sys.modules`
is exactly what the probe reads.
"""

from __future__ import annotations

import sys
import types

import pytest

from modules import directml_device
from modules import ort_directml


class _FakeOrt(types.ModuleType):
    """Stands in for the onnxruntime package."""

    def __init__(self, providers, version="1.24.4"):
        super().__init__("onnxruntime")
        self.__version__ = version
        self._providers = list(providers)
        self.sessions = []

    def get_available_providers(self):
        return list(self._providers)

    def InferenceSession(self, path, providers=None, **kwargs):   # noqa: N802
        session = types.SimpleNamespace(path=path, providers=providers,
                                        get_providers=lambda: [
                                            p[0] if isinstance(p, tuple) else p
                                            for p in (providers or [])])
        self.sessions.append(session)
        return session


@pytest.fixture(autouse=True)
def _clean_probe(monkeypatch):
    """Every test starts with no cached answer and DirectML on."""
    monkeypatch.delenv(directml_device.MODE_ENV, raising=False)
    monkeypatch.delenv(ort_directml.DEVICE_ENV, raising=False)
    directml_device.set_mode(None)
    ort_directml.reset_probe_cache()
    yield
    directml_device.set_mode(None)
    ort_directml.reset_probe_cache()


def _install(monkeypatch, ort):
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    ort_directml.reset_probe_cache()
    return ort


class TestTheProbe:
    def test_the_directml_build_is_available(self, monkeypatch):
        _install(monkeypatch, _FakeOrt(["DmlExecutionProvider", "CPUExecutionProvider"]))

        probe = ort_directml.probe()

        assert probe.available
        assert probe.reason is None
        assert probe.version == "1.24.4"

    def test_the_plain_build_says_which_one_is_installed(self, monkeypatch):
        """'onnxruntime is not installed' would be a lie here, and would send
        the user to install the package they already have."""
        _install(monkeypatch, _FakeOrt(["CPUExecutionProvider"]))

        reason = ort_directml.unavailable_reason()

        assert not ort_directml.available()
        assert "CPUExecutionProvider" in reason
        assert "onnxruntime-directml" in reason

    def test_a_missing_package_says_so(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "onnxruntime", raising=False)
        monkeypatch.setattr(ort_directml, "_import_onnxruntime",
                            lambda: (_ for _ in ()).throw(ModuleNotFoundError("no")))
        ort_directml.reset_probe_cache()

        reason = ort_directml.unavailable_reason()

        assert "not installed" in reason
        assert "ModuleNotFoundError" in reason

    def test_the_user_switch_turns_both_runtimes_off(self, monkeypatch):
        """One DirectML switch. A user who turned it off does not mean 'off for
        torch, on for ONNX'."""
        _install(monkeypatch, _FakeOrt(["DmlExecutionProvider", "CPUExecutionProvider"]))
        directml_device.set_mode("off")
        ort_directml.reset_probe_cache()

        assert not ort_directml.available()
        assert directml_device.MODE_ENV in ort_directml.unavailable_reason()

    def test_the_answer_is_cached(self, monkeypatch):
        ort = _install(monkeypatch, _FakeOrt(["DmlExecutionProvider"]))
        calls = []
        ort.get_available_providers = lambda: calls.append(1) or ["DmlExecutionProvider"]
        ort_directml.reset_probe_cache()

        ort_directml.probe()
        ort_directml.probe()

        assert len(calls) == 1


class TestProviderList:
    def test_directml_comes_first_and_cpu_last(self, monkeypatch):
        """CPU always stays on the list: DirectML covers a subset of the
        operators, and a per-node fallback beats a failed run."""
        _install(monkeypatch, _FakeOrt(["DmlExecutionProvider", "CPUExecutionProvider"]))

        providers = ort_directml.providers()

        assert providers[0][0] == "DmlExecutionProvider"
        assert providers[-1] == "CPUExecutionProvider"

    def test_the_adapter_index_is_settable(self, monkeypatch):
        _install(monkeypatch, _FakeOrt(["DmlExecutionProvider", "CPUExecutionProvider"]))
        monkeypatch.setenv(ort_directml.DEVICE_ENV, "1")

        assert ort_directml.providers()[0][1] == {"device_id": 1}

    def test_a_nonsense_adapter_index_falls_back_to_zero(self, monkeypatch):
        monkeypatch.setenv(ort_directml.DEVICE_ENV, "second one")

        assert ort_directml.device_id() == 0

    def test_without_directml_the_list_is_cpu_only(self, monkeypatch):
        _install(monkeypatch, _FakeOrt(["CPUExecutionProvider"]))

        assert ort_directml.providers() == ["CPUExecutionProvider"]


class TestSessions:
    def test_the_session_is_built_on_the_provider_list(self, monkeypatch):
        ort = _install(monkeypatch, _FakeOrt(["DmlExecutionProvider", "CPUExecutionProvider"]))

        session = ort_directml.session("model.onnx")

        assert ort.sessions == [session]
        assert session.providers[0][0] == "DmlExecutionProvider"

    def test_the_backend_is_read_back_off_the_session(self, monkeypatch):
        """ORT drops a provider it cannot initialise without saying so, so a run
        that quietly landed on the processor has to be able to admit it."""
        _install(monkeypatch, _FakeOrt(["DmlExecutionProvider", "CPUExecutionProvider"]))
        session = ort_directml.session("model.onnx")
        session.get_providers = lambda: ["CPUExecutionProvider"]

        assert ort_directml.session_backend(session) == "CPUExecutionProvider"

    def test_a_session_that_cannot_answer_is_not_an_error(self, monkeypatch):
        broken = types.SimpleNamespace(
            get_providers=lambda: (_ for _ in ()).throw(RuntimeError("gone")))

        assert ort_directml.session_backend(broken) == "unknown"
