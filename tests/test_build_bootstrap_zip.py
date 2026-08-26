"""Bootstrap zip builder for Windows one-click install.

The installer decides which release to fetch, so the version it names has to
follow version.py rather than be remembered. It did not: the committed
config.json still said 0.9.0 several releases later, and that value is what the
installer falls back to whenever the GitHub API cannot be reached — so an
offline or rate-limited run quietly installed a version that was months old.
These tests fail the moment the two disagree again, and the failure says which
command puts it right.
"""
from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "build_bootstrap_zip.py"
BOOTSTRAP = ROOT / "packaging" / "bootstrap"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("build_bootstrap_zip", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def app_version():
    import version

    return version.__version__


def test_build_free_bootstrap_zip(mod, tmp_path: Path) -> None:
    out = tmp_path / "VideoHighlighter-Windows-Setup.zip"
    mod.build_zip(edition="free", tag="0.9.0", out=out)

    assert out.is_file()
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
        config = json.loads(zf.read("config.json"))
    assert names == {
        "Install-VideoHighlighter.bat",
        "Install-VideoHighlighter.ps1",
        "config.json",
    }
    assert config["edition"] == "Free"
    assert config["use_latest"] is True


class TestTheTagFollowsVersionPy:
    def test_free_defaults_to_the_bare_version(self, mod, app_version):
        assert mod.default_tag("free") == app_version

    def test_pro_defaults_to_the_version_plus_edition(self, mod, app_version):
        # Matches the slug the release workflow builds from version.py.
        assert mod.default_tag("pro") == f"{app_version}-Pro"

    def test_an_explicit_tag_still_wins(self, mod, tmp_path: Path):
        # CI passes the resolved release slug; nothing here should override it.
        out = tmp_path / "setup.zip"
        mod.build_zip(edition="free", tag="1.2.3", out=out)
        with zipfile.ZipFile(out) as zf:
            config = json.loads(zf.read("config.json"))
        assert config["tag"] == "1.2.3"


class TestTheCommittedConfigsAreCurrent:
    """What someone gets running the installer straight from a checkout."""

    @pytest.mark.parametrize("edition", ["free", "pro"])
    def test_the_pinned_tag_matches_this_checkout(self, mod, edition, app_version):
        path = mod.CONFIG_PATHS[edition]
        if not path.exists():
            pytest.skip(f"{path.name} is not part of this edition")

        config = json.loads(path.read_text(encoding="utf-8"))
        expected = mod.default_tag(edition)
        assert config["tag"] == expected, (
            f"{path.name} pins {config['tag']!r} but version.py says "
            f"{app_version!r}. Run: python tools/build_bootstrap_zip.py "
            f"--edition {edition} --write-config"
        )

    @pytest.mark.parametrize("edition", ["free", "pro"])
    def test_the_asset_names_and_url_carry_that_tag(self, mod, edition):
        path = mod.CONFIG_PATHS[edition]
        if not path.exists():
            pytest.skip(f"{path.name} is not part of this edition")

        config = json.loads(path.read_text(encoding="utf-8"))
        tag = config["tag"]
        # A tag that agreed while the asset names did not would still send the
        # fallback path at files that do not exist.
        assert config["base_url"].endswith(f"/{tag}")
        assert config["assets"], "no fallback assets listed"
        for name in config["assets"]:
            assert tag in name, f"{name} does not name the pinned tag {tag}"

    @pytest.mark.parametrize("edition", ["free", "pro"])
    def test_the_asset_names_match_the_pattern_used_online(self, mod, edition):
        import re

        path = mod.CONFIG_PATHS[edition]
        if not path.exists():
            pytest.skip(f"{path.name} is not part of this edition")

        config = json.loads(path.read_text(encoding="utf-8"))
        # The API path selects assets by this regex and the offline path uses
        # the listed names. If the two ever describe different files, which one
        # you get depends on whether GitHub answered.
        pattern = re.compile(config["asset_pattern"])
        for name in config["assets"]:
            assert pattern.match(name), f"{name} would not be picked up online"


class TestWriteConfigKeepsWhatIsNotDerived:
    def test_hand_written_notes_survive_a_rewrite(self, mod, tmp_path, monkeypatch):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"notes": "run it like this"}), encoding="utf-8")
        monkeypatch.setitem(mod.CONFIG_PATHS, "free", path)

        mod.write_config(edition="free", tag="9.9.9")

        written = json.loads(path.read_text(encoding="utf-8"))
        assert written["tag"] == "9.9.9"
        assert written["notes"] == "run it like this"

    def test_it_refuses_a_config_that_is_not_there(self, mod, tmp_path, monkeypatch):
        monkeypatch.setitem(mod.CONFIG_PATHS, "pro", tmp_path / "nope.json")
        with pytest.raises(SystemExit):
            mod.write_config(edition="pro", tag="9.9.9")
