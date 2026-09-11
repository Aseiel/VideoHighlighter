"""The CA bundle the frozen app falls back to.

A macOS build verifies HTTPS against an OpenSSL directory that python.org's
installer script populates and a frozen app never does, so every call failed
with CERTIFICATE_VERIFY_FAILED. On 0.11.0 that turned the update check into
"no manifest", which a user reads as "I am up to date".
"""

from __future__ import annotations

import sys
import types

import pytest

from modules import https_certs


@pytest.fixture(autouse=True)
def _uncached():
    https_certs.reset_cache()
    yield
    https_certs.reset_cache()


class TestTheContext:
    def test_it_is_built_on_certifis_bundle(self, monkeypatch, tmp_path):
        bundle = tmp_path / "cacert.pem"
        bundle.write_text("")
        asked = {}

        fake_certifi = types.ModuleType("certifi")
        fake_certifi.where = lambda: str(bundle)
        fake_ssl = types.ModuleType("ssl")

        def create_default_context(cafile=None):
            asked["cafile"] = cafile
            return "a context"

        fake_ssl.create_default_context = create_default_context
        monkeypatch.setitem(sys.modules, "certifi", fake_certifi)
        monkeypatch.setitem(sys.modules, "ssl", fake_ssl)

        assert https_certs.context() == "a context"
        assert asked["cafile"] == str(bundle)

    def test_no_certifi_means_urllibs_own_behaviour(self, monkeypatch):
        """Windows reads the system store and is fine; None is the honest
        answer there rather than a broken context."""
        monkeypatch.setitem(sys.modules, "certifi", None)

        assert https_certs.context() is None
        assert https_certs.opener_kwargs() == {}

    def test_a_broken_bundle_does_not_stop_the_launch(self, monkeypatch, capsys):
        fake_certifi = types.ModuleType("certifi")
        fake_certifi.where = lambda: "/nowhere/cacert.pem"
        fake_ssl = types.ModuleType("ssl")

        def explode(cafile=None):
            raise OSError("no such file")

        fake_ssl.create_default_context = explode
        monkeypatch.setitem(sys.modules, "certifi", fake_certifi)
        monkeypatch.setitem(sys.modules, "ssl", fake_ssl)

        assert https_certs.context() is None
        assert "unusable" in capsys.readouterr().out

    def test_the_context_is_built_once(self, monkeypatch):
        """The update check runs at launch, and parsing the CA file is not
        free."""
        calls = []
        fake_certifi = types.ModuleType("certifi")
        fake_certifi.where = lambda: "bundle.pem"
        fake_ssl = types.ModuleType("ssl")
        fake_ssl.create_default_context = lambda cafile=None: calls.append(cafile) or "ctx"
        monkeypatch.setitem(sys.modules, "certifi", fake_certifi)
        monkeypatch.setitem(sys.modules, "ssl", fake_ssl)

        https_certs.context()
        https_certs.context()

        assert len(calls) == 1


class TestOpenerKwargs:
    def test_it_is_what_urlopen_takes(self, monkeypatch):
        monkeypatch.setattr(https_certs, "context", lambda: "ctx")

        assert https_certs.opener_kwargs() == {"context": "ctx"}

    def test_real_certifi_gives_a_usable_context(self):
        """Not a mock: if certifi is installed, the file it names must parse."""
        pytest.importorskip("certifi", reason="certifi not installed here")
        import ssl

        ctx = https_certs.context()

        assert isinstance(ctx, ssl.SSLContext)
        assert ctx.get_ca_certs(), "the bundle loaded no certificates"
