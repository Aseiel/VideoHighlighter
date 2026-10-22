"""Tests for the download mirror's bookkeeping (tools/mirror_release.py).

mirror-release.yaml copies the bytes; what it writes beside them, and what
prune-updates.yaml deletes, is decided here. What's pinned down: the layout
every downloader derives from, a signed channel entry never being replaced by
a download-only one, and packs tags never being pruned.
"""
from __future__ import annotations

import json

import pytest

from tools import mirror_release as mr

SHA = "ab" * 32


def _index(tag="0.14.0", names=("00-VideoHighlighter-Windows-Setup.exe",
                                 "VideoHighlighter-Windows-0.14.0.7z.001")):
    return mr.build_index(tag, [(n, 2**20, SHA) for n in names])


def test_the_layout_every_downloader_derives():
    assert mr.mirror_base("https://dl.example/", "packs-torch-2.7.1-r1") == \
        "https://dl.example/downloads/packs-torch-2.7.1-r1"
    assert mr.mirror_base("", "0.14.0") == ""


def test_the_lock_writer_and_bootstrap_spell_the_same_layout(tmp_path):
    from tools import build_bootstrap_zip, write_packs_lock

    packs = tmp_path / "packs.json"
    packs.write_text(json.dumps({"tag": "packs-x", "packs": [
        {"name": "torch-cpu", "asset": "a.7z", "bytes": 5, "sha256": SHA}]}), encoding="utf-8")
    out = tmp_path / "lock.json"
    assert write_packs_lock.main(["--base", "https://github.com/o/r/releases/download/packs-x",
                                  "--packs-json", str(packs), "--mirror-host", "https://dl.example/",
                                  "--out", str(out)]) == 0
    assert json.loads(out.read_text())["mirrors"] == [mr.mirror_base("https://dl.example", "packs-x")]

    config = build_bootstrap_zip.make_config(edition="free", tag="0.14.0",
                                             mirror_host="https://dl.example")
    assert f"{config['mirror']}/0.14.0" == mr.mirror_base("https://dl.example", "0.14.0")


def test_no_mirror_host_means_no_mirrors_in_the_lock(tmp_path):
    from tools import write_packs_lock

    packs = tmp_path / "packs.json"
    packs.write_text(json.dumps({"packs": [
        {"name": "torch-cpu", "asset": "a.7z", "bytes": 5, "sha256": SHA}]}), encoding="utf-8")
    out = tmp_path / "lock.json"
    write_packs_lock.main(["--base", "https://h/packs-x", "--packs-json", str(packs),
                           "--out", str(out)])
    assert "mirrors" not in json.loads(out.read_text())


def test_index_is_sorted_and_refuses_what_is_not_a_hash():
    index = _index(names=("b", "a"))
    assert [a["name"] for a in index["assets"]] == ["a", "b"]
    with pytest.raises(ValueError):
        mr.build_index("0.14.0", [("a", 1, "nope")])
    with pytest.raises(ValueError):
        mr.build_index("0.14.0", [])


def test_only_fixed_names_are_copied_to_latest():
    assert mr.fixed_names(_index()) == ["00-VideoHighlighter-Windows-Setup.exe"]


def test_the_page_links_installers_beside_it_and_the_rest_to_their_tag():
    page = mr.latest_page(_index(names=("00-Setup.exe", "a<b>.7z.001")),
                          notes_url="https://github.com/o/r/releases/tag/0.14.0")
    assert 'href="00-Setup.exe"' in page
    assert 'href="../0.14.0/a&lt;b&gt;.7z.001"' in page
    assert "a<b>" not in page
    assert SHA in page and "releases/tag/0.14.0" in page


class TestTheChannel:
    def test_a_newer_release_is_announced_for_download_only(self):
        payload = mr.download_channel(_index("0.14.0"), {"version": "0.13.3", "manifests": {
            "windows": "https://dl.example/releases/free/windows/0.13.3/manifest.json"}},
            notes_url="n", download_url="https://dl.example/downloads/latest/index.html",
            today="2026-10-22")
        assert payload == {"version": "0.14.0", "date": "2026-10-22", "notes": "",
                           "notes_url": "n",
                           "download_url": "https://dl.example/downloads/latest/index.html",
                           "manifests": {}}

    def test_an_empty_channel_is_written(self):
        assert mr.download_channel(_index("0.14.0"), {})["version"] == "0.14.0"

    def test_the_signed_entry_for_the_same_release_is_left_alone(self):
        signed = {"version": "0.14.0", "manifests": {"windows": "m"}}
        assert mr.download_channel(_index("0.14.0"), signed) is None

    def test_it_never_moves_backwards(self):
        assert mr.download_channel(_index("0.13.3"), {"version": "0.14.0"}) is None

    def test_cli_writes_only_when_it_should(self, tmp_path):
        index = tmp_path / "index.json"
        index.write_text(json.dumps(_index("0.14.0")))
        previous = tmp_path / "previous.json"
        out = tmp_path / "channel.json"
        previous.write_text(json.dumps({"version": "0.14.0"}))
        out.write_text("stale")
        args = ["channel", "--index", str(index), "--previous", str(previous), "--out", str(out)]
        assert mr.main(args) == 0 and not out.exists()
        previous.write_text("{}")
        assert mr.main(args) == 0
        assert json.loads(out.read_text())["version"] == "0.14.0"


def test_signing_later_keeps_the_mirror_link():
    from tools import publish_release

    manifest = {"version": "0.14.0", "platform": "windows", "base_url": "https://dl.example"}
    announced = mr.download_channel(_index("0.14.0"), {}, download_url="https://dl.example/page")
    signed = publish_release.build_channel(manifest, "m", previous=announced)
    assert signed["download_url"] == "https://dl.example/page"
    assert signed["manifests"] == {"windows": "m"}
    # An older announcement's link is not carried onto a newer release.
    newer = dict(manifest, version="0.14.1")
    assert publish_release.build_channel(newer, "m", previous=announced)["download_url"] == ""


class TestPrune:
    def test_keeps_the_newest_and_what_latest_names(self):
        tags = ["0.12.0", "0.13.0", "0.13.10", "0.13.9", "0.14.0", "latest"]
        assert mr.tags_to_delete(tags, keep=2) == ["0.12.0", "0.13.0", "0.13.9"]
        assert mr.tags_to_delete(tags, keep=2, latest_tags=["0.13.0"]) == ["0.12.0", "0.13.9"]

    def test_packs_tags_are_never_deleted(self):
        assert mr.tags_to_delete(["packs-torch-2.6.0", "packs-torch-2.7.1-r1", "0.14.0"],
                                 keep=1) == []

    def test_cli_reads_the_listing_and_latest(self, tmp_path, capsys):
        tags = tmp_path / "tags.json"
        tags.write_text(json.dumps([f"downloads/{t}/" for t in
                                    ("0.13.0", "0.13.1", "0.14.0", "latest", "packs-torch-2.7.1")]))
        latest = tmp_path / "latest"
        latest.mkdir()
        (latest / "free.json").write_text(json.dumps({"tag": "0.13.0"}))
        assert mr.main(["prune", "--tags", str(tags), "--latest", str(latest), "--keep", "1"]) == 0
        assert capsys.readouterr().out.split() == ["0.13.1"]

    def test_an_empty_bucket_deletes_nothing(self, tmp_path, capsys):
        tags = tmp_path / "tags.json"
        tags.write_text("null")
        assert mr.main(["prune", "--tags", str(tags), "--keep", "3"]) == 0
        assert capsys.readouterr().out == ""


def test_the_mac_lock_keeps_only_the_packs_it_names(tmp_path):
    # The PyTorch packs are Windows builds; a Mac must never be offered them.
    from tools import write_packs_lock

    packs = tmp_path / "packs.json"
    packs.write_text(json.dumps({"tag": "packs-x", "packs": [
        {"name": n, "asset": f"{n}.7z", "bytes": 5, "sha256": SHA}
        for n in ("torch-cpu", "torch-cu128", "models-frame-encoder", "models-clip")]}),
        encoding="utf-8")
    out = tmp_path / "lock.json"
    assert write_packs_lock.main(["--base", "https://h/packs-x", "--packs-json", str(packs),
                                  "--names", "models-frame-encoder", "--out", str(out)]) == 0
    assert [p["name"] for p in json.loads(out.read_text())["packs"]] == ["models-frame-encoder"]
