"""
Tests for modules.gopro_ingest against a synthetic card on disk.

The card is built as real directories and files (a few KB each) rather than
mocked, because every behaviour worth testing here is a filesystem behaviour:
that detection keys on layout instead of a drive label, that GoPro's
chapter-before-file-number naming is sorted back into recording order, that an
interrupted copy cannot leave a short file passing as complete, and that a
second run copies nothing.

No ffprobe is required: ingest() is called with probe=False except where a
test is specifically about probing.
"""

from __future__ import annotations

import json
import os
import time

import pytest

from modules.gopro_ingest import (
    CopyCancelled,
    GoProCard,
    find_gopro_cards,
    ingest,
    read_manifest,
    scan_card,
    suggest_folder_name,
    write_manifest,
)

# Shaped exactly like a real MISC/version.txt (including the leading-comma
# formatting the camera writes), with invented identifiers. Nothing here comes
# off anyone's actual camera.
VERSION_JSON = """{
"info version":"2.0"
,"firmware version":"H24.01.02.04.00"
,"wifi mac":"000000000000"
,"camera type":"HERO13 Black"
,"camera serial number":"C0000000000000"
}"""


def _make_card(root, names, *, version_txt: str | None = VERSION_JSON,
               media_dir="100GOPRO", size=2048):
    """Write a minimal GoPro card layout and return its root path."""
    media = root / "DCIM" / media_dir
    media.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(names):
        # Distinct content per file so hash verification is meaningful.
        (media / name).write_bytes(bytes([i % 251]) * size)
    if version_txt is not None:
        misc = root / "MISC"
        misc.mkdir(exist_ok=True)
        (misc / "version.txt").write_text(version_txt, encoding="utf-8")
    return str(root)


def _card_for(root) -> GoProCard:
    """The synthetic card at ``root``, ignoring any real card mounted on the
    machine running the tests. Picking cards[0] would silently pass or fail
    against whatever happens to be in the developer's card reader.
    """
    cards = find_gopro_cards(extra_roots=[str(root)], scan_mounts=False)
    assert cards, "expected the synthetic card to be detected"
    return cards[0]


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def test_detects_card_by_layout_and_reads_camera_metadata(tmp_path):
    _make_card(tmp_path, ["GX013762.MP4", "GX013763.MP4"])
    card = _card_for(tmp_path)

    assert card.camera_type == "HERO13 Black"
    assert card.firmware == "H24.01.02.04.00"
    assert card.serial == "C0000000000000"
    assert card.file_count == 2
    assert card.total_bytes == 4096


def test_card_without_version_file_is_still_detected(tmp_path):
    """A card whose MISC/version.txt is missing must still ingest; only the
    provenance fields go empty."""
    _make_card(tmp_path, ["GX013762.MP4"], version_txt=None)
    card = _card_for(tmp_path)

    assert card.camera_type == ""
    assert card.file_count == 1
    assert card.label.startswith("GoPro card")


def test_malformed_version_file_does_not_break_detection(tmp_path):
    _make_card(tmp_path, ["GX013762.MP4"], version_txt="{not json at all")
    card = _card_for(tmp_path)

    assert card.file_count == 1
    assert card.camera_type == ""


def test_non_gopro_folder_is_not_a_card(tmp_path):
    """A DCIM folder that is not a GoPro media dir must not match — this is
    what stops an ordinary camera or phone card being treated as GoPro."""
    (tmp_path / "DCIM" / "100CANON").mkdir(parents=True)
    (tmp_path / "DCIM" / "100CANON" / "IMG_0001.MP4").write_bytes(b"x" * 16)

    assert find_gopro_cards(extra_roots=[str(tmp_path)], scan_mounts=False) == []


def test_gopro_dir_with_no_parseable_media_is_not_a_card(tmp_path):
    (tmp_path / "DCIM" / "100GOPRO").mkdir(parents=True)
    (tmp_path / "DCIM" / "100GOPRO" / "notes.txt").write_text("hi")

    assert find_gopro_cards(extra_roots=[str(tmp_path)], scan_mounts=False) == []


def test_multiple_media_dirs_are_all_scanned(tmp_path):
    _make_card(tmp_path, ["GX013762.MP4"], media_dir="100GOPRO")
    _make_card(tmp_path, ["GX013900.MP4"], media_dir="101GOPRO")
    card = _card_for(tmp_path)

    assert card.file_count == 2
    assert len(card.media_dirs) == 2


# ---------------------------------------------------------------------------
# Naming / ordering — the reason this module exists
# ---------------------------------------------------------------------------

def test_chapters_group_into_takes_in_recording_order(tmp_path):
    """GoPro puts the chapter before the file number, so alphabetical order
    interleaves separate recordings. Takes must come back as 0527 then 0528,
    each with its chapters in sequence."""
    _make_card(tmp_path, [
        "GH010527.MP4", "GH010528.MP4", "GH020527.MP4", "GH020528.MP4",
    ])
    takes = scan_card(_card_for(tmp_path))

    assert [t.file_number for t in takes] == [527, 528]
    assert [c.name for c in takes[0].clips] == ["GH010527.MP4", "GH020527.MP4"]
    assert [c.name for c in takes[1].clips] == ["GH010528.MP4", "GH020528.MP4"]
    assert all(t.is_chaptered for t in takes)


def test_single_chapter_files_are_separate_takes(tmp_path):
    """The real card in this project: 21 files all at chapter 01, which are 21
    distinct recordings rather than one long take."""
    _make_card(tmp_path, [f"GX01{n}.MP4" for n in (3762, 3763, 3764)])
    takes = scan_card(_card_for(tmp_path))

    assert [t.file_number for t in takes] == [3762, 3763, 3764]
    assert not any(t.is_chaptered for t in takes)


def test_codec_prefix_is_recorded(tmp_path):
    _make_card(tmp_path, ["GX013762.MP4", "GH013763.MP4"])
    takes = scan_card(_card_for(tmp_path))

    assert takes[0].clips[0].codec == "HEVC"
    assert takes[1].clips[0].codec == "AVC"


def test_low_res_proxies_and_unknown_names_are_ignored(tmp_path):
    """GL files are LRV proxies of clips already being copied, and a name that
    does not parse has no reliable take — guessing would reorder footage."""
    _make_card(tmp_path, ["GX013762.MP4", "GL013762.MP4", "randomclip.MP4"])
    card = _card_for(tmp_path)
    takes = scan_card(card)

    assert card.file_count == 1
    assert [c.name for t in takes for c in t.clips] == ["GX013762.MP4"]


def test_suggested_folder_name_uses_shoot_date_not_today(tmp_path):
    """Re-ingesting an old card must file footage under the day it was shot."""
    _make_card(tmp_path, ["GX013762.MP4"])
    card = _card_for(tmp_path)
    takes = scan_card(card)
    # Built from a local date rather than a hardcoded epoch: the constant would
    # name a different day in another timezone, and the test would be asserting
    # the tester's offset instead of the behaviour.
    shot = time.mktime((2026, 8, 8, 12, 0, 0, 0, 0, -1))
    for clip in takes[0].clips:
        clip.mtime = shot

    name = suggest_folder_name(card, takes)

    assert name.startswith("2026-08-08")
    assert "HERO13-Black" in name


# ---------------------------------------------------------------------------
# Copying
# ---------------------------------------------------------------------------

def test_ingest_copies_every_file_and_reports_paths(tmp_path):
    src = tmp_path / "card"
    _make_card(src, ["GX013762.MP4", "GX013763.MP4"])
    dest = tmp_path / "out"

    result = ingest(_card_for(src), str(dest), folder_name="shoot",
                    probe=False, log_fn=lambda *_: None)

    assert len(result.paths) == 2
    assert all(os.path.exists(p) for p in result.paths)
    assert result.copied_bytes == 4096
    assert result.errors == []
    assert [os.path.basename(p) for p in result.paths] == [
        "GX013762.MP4", "GX013763.MP4"]


def test_copied_bytes_match_the_source_exactly(tmp_path):
    src = tmp_path / "card"
    _make_card(src, ["GX013762.MP4"], size=100_000)
    dest = tmp_path / "out"

    result = ingest(_card_for(src), str(dest), folder_name="shoot",
                    probe=False, log_fn=lambda *_: None)

    original = (src / "DCIM" / "100GOPRO" / "GX013762.MP4").read_bytes()
    assert open(result.paths[0], "rb").read() == original


def test_second_run_skips_files_already_present(tmp_path):
    """Re-running after a failure must be cheap — the card is usually still in
    the slot."""
    src = tmp_path / "card"
    _make_card(src, ["GX013762.MP4", "GX013763.MP4"])
    dest = tmp_path / "out"
    common = dict(folder_name="shoot", probe=False, log_fn=lambda *_: None)

    ingest(_card_for(src), str(dest), **common)
    second = ingest(_card_for(src), str(dest), **common)

    assert second.copied_bytes == 0
    assert second.skipped_bytes == 4096
    assert all(f.skipped for f in second.files)


def test_truncated_destination_is_recopied(tmp_path):
    """A short file from an interrupted run must not be mistaken for a good
    copy just because the name is right."""
    src = tmp_path / "card"
    _make_card(src, ["GX013762.MP4"])
    dest = tmp_path / "out"
    common = dict(folder_name="shoot", probe=False, log_fn=lambda *_: None)
    ingest(_card_for(src), str(dest), **common)

    landed = dest / "shoot" / "GX013762.MP4"
    landed.write_bytes(b"short")

    result = ingest(_card_for(src), str(dest), **common)

    assert result.copied_bytes == 2048
    assert landed.stat().st_size == 2048


def test_hash_verification_recopies_a_corrupted_file(tmp_path):
    """Same size, different bytes — only hash verification catches this."""
    src = tmp_path / "card"
    _make_card(src, ["GX013762.MP4"])
    dest = tmp_path / "out"
    common = dict(folder_name="shoot", probe=False, log_fn=lambda *_: None)
    ingest(_card_for(src), str(dest), **common)

    landed = dest / "shoot" / "GX013762.MP4"
    landed.write_bytes(b"\xff" * 2048)

    result = ingest(_card_for(src), str(dest), verify="hash", **common)

    assert result.copied_bytes == 2048
    assert landed.read_bytes() == (src / "DCIM" / "100GOPRO" / "GX013762.MP4").read_bytes()


def test_cancel_stops_the_run_and_leaves_no_partial_file(tmp_path):
    src = tmp_path / "card"
    _make_card(src, ["GX013762.MP4", "GX013763.MP4"], size=64 * 1024)
    dest = tmp_path / "out"

    with pytest.raises(CopyCancelled):
        ingest(_card_for(src), str(dest), folder_name="shoot", probe=False,
               log_fn=lambda *_: None, cancel_check=lambda: True)

    landed = dest / "shoot"
    leftovers = list(landed.glob("*.part")) if landed.exists() else []
    assert leftovers == []


def test_unreadable_file_is_recorded_but_others_still_copy(tmp_path, monkeypatch):
    """One bad clip is a bad sector, not a reason to abandon the rest."""
    src = tmp_path / "card"
    _make_card(src, ["GX013762.MP4", "GX013763.MP4"])
    dest = tmp_path / "out"

    import modules.gopro_ingest as gi
    real_copy = gi._copy_one

    def flaky(source, target, size, verify, **kwargs):
        if source.endswith("GX013762.MP4"):
            raise OSError("simulated read error")
        return real_copy(source, target, size, verify, **kwargs)

    monkeypatch.setattr(gi, "_copy_one", flaky)

    result = ingest(_card_for(src), str(dest), folder_name="shoot",
                    probe=False, log_fn=lambda *_: None)

    assert len(result.errors) == 1
    assert "GX013762.MP4" in result.errors[0]
    assert [os.path.basename(p) for p in result.paths] == ["GX013763.MP4"]


def test_progress_reaches_the_total_even_when_a_file_fails(tmp_path, monkeypatch):
    """A progress bar that stops at 60% because one clip failed is a bug
    report waiting to happen."""
    src = tmp_path / "card"
    _make_card(src, ["GX013762.MP4", "GX013763.MP4"])
    dest = tmp_path / "out"

    import modules.gopro_ingest as gi
    real_copy = gi._copy_one

    def flaky(source, target, size, verify, **kwargs):
        if source.endswith("GX013762.MP4"):
            raise OSError("simulated read error")
        return real_copy(source, target, size, verify, **kwargs)

    monkeypatch.setattr(gi, "_copy_one", flaky)
    seen: list[tuple[int, int]] = []

    ingest(_card_for(src), str(dest), folder_name="shoot", probe=False,
           log_fn=lambda *_: None,
           progress_fn=lambda done, total, name: seen.append((done, total)))

    assert seen
    done, total = seen[-1]
    assert done == total == 4096


def test_progress_is_reported_in_byte_increments(tmp_path):
    src = tmp_path / "card"
    _make_card(src, ["GX013762.MP4"], size=32 * 1024)
    dest = tmp_path / "out"
    seen: list[tuple[int, int, str]] = []

    ingest(_card_for(src), str(dest), folder_name="shoot", probe=False,
           log_fn=lambda *_: None,
           progress_fn=lambda done, total, name: seen.append((done, total, name)))

    assert seen[-1][0] == seen[-1][1] == 32 * 1024
    assert seen[-1][2] == "GX013762.MP4"


def test_nothing_is_deleted_from_the_card(tmp_path):
    src = tmp_path / "card"
    _make_card(src, ["GX013762.MP4", "GX013763.MP4"])
    dest = tmp_path / "out"
    before = sorted(os.listdir(src / "DCIM" / "100GOPRO"))

    ingest(_card_for(src), str(dest), folder_name="shoot", probe=False,
           log_fn=lambda *_: None)

    assert sorted(os.listdir(src / "DCIM" / "100GOPRO")) == before


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def test_manifest_records_takes_as_groups(tmp_path):
    src = tmp_path / "card"
    _make_card(src, ["GH010527.MP4", "GH020527.MP4", "GH010528.MP4"])
    dest = tmp_path / "out"

    result = ingest(_card_for(src), str(dest), folder_name="shoot",
                    probe=False, log_fn=lambda *_: None)
    path = write_manifest(result)
    data = read_manifest(path)

    assert data["version"] == 1
    assert len(data["takes"]) == 2
    assert len(data["takes"][0]) == 2      # 0527 kept together as one take
    assert len(data["takes"][1]) == 1
    assert data["card"]["camera_type"] == "HERO13 Black"
    assert len(data["files"]) == 3


def test_manifest_lands_next_to_the_footage_by_default(tmp_path):
    src = tmp_path / "card"
    _make_card(src, ["GX013762.MP4"])
    dest = tmp_path / "out"

    result = ingest(_card_for(src), str(dest), folder_name="shoot",
                    probe=False, log_fn=lambda *_: None)
    path = write_manifest(result)

    assert os.path.dirname(path) == str(dest / "shoot")
    assert os.path.basename(path) == "ingest.json"


def test_reading_an_unknown_manifest_version_raises(tmp_path):
    path = tmp_path / "ingest.json"
    path.write_text(json.dumps({"version": 99}), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported manifest version"):
        read_manifest(str(path))
