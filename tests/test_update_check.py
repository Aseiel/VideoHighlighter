"""Tests for the update notification check.

No network: ``check_for_update`` takes an injectable transport and the state
file is monkeypatched into a temp dir. What's pinned down is the decision layer
— version ordering, the throttle, the skip list, and the rule that every
failure is silent — because those decide whether a user is nagged, misled, or
left stale.
"""
from __future__ import annotations

import datetime as dt

import pytest

from modules import update_check


def _manifest(version="0.9.1", **extra):
    payload = {
        "version": version,
        "date": "2026-08-10",
        "notes": "Lemon Squeezy license keys.",
        "notes_url": "https://example.invalid/notes",
    }
    payload.update(extra)
    return payload


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(update_check, "state_path",
                        lambda: str(tmp_path / "update_state.json"))
    return tmp_path


def _serving(payload):
    def transport(url):
        return payload
    return transport


def _failing(exc=OSError("no route to host")):
    def transport(url):
        raise exc
    return transport


# --- version ordering ------------------------------------------------------

@pytest.mark.parametrize("candidate,current,expected", [
    ("0.9.1", "0.9.0", True),
    ("0.10.0", "0.9.9", True),      # not string ordering
    ("1.0.0", "0.9.9", True),
    ("0.9.0", "0.9.0", False),
    ("0.8.3", "0.9.0", False),
    ("0.9", "0.9.0", False),        # padded, not shorter-is-smaller
    ("0.9.0.1", "0.9.0", True),
    ("", "0.9.0", False),           # garbage never announces an update
    ("not-a-version", "0.9.0", False),
])
def test_is_newer(candidate, current, expected):
    assert update_check.is_newer(candidate, current) is expected


# --- the happy path --------------------------------------------------------

def test_reports_a_newer_version(state_dir):
    info = update_check.check_for_update(
        current_version="0.9.0", transport=_serving(_manifest()))
    assert info is not None
    assert info.version == "0.9.1"
    assert "0.9.1" in info.headline


def test_silent_when_up_to_date(state_dir):
    assert update_check.check_for_update(
        current_version="0.9.1", transport=_serving(_manifest("0.9.1"))) is None


def test_download_url_defaults_by_edition(state_dir):
    info = update_check.check_for_update(
        current_version="0.9.0", transport=_serving(_manifest()))
    assert info.download_url == update_check._DEFAULT_LANDING[
        update_check._channel()]


def test_manifest_can_override_the_download_url(state_dir):
    info = update_check.check_for_update(
        current_version="0.9.0",
        transport=_serving(_manifest(download_url="https://example.invalid/x")))
    assert info.download_url == "https://example.invalid/x"


# --- failure is always silent ---------------------------------------------

def test_network_failure_is_silent(state_dir):
    assert update_check.check_for_update(
        current_version="0.9.0", transport=_failing()) is None


def test_malformed_manifest_is_silent(state_dir):
    assert update_check.check_for_update(
        current_version="0.9.0", transport=_serving(["not", "a", "dict"])) is None


def test_manifest_without_a_version_is_silent(state_dir):
    assert update_check.check_for_update(
        current_version="0.9.0", transport=_serving({"notes": "hi"})) is None


# --- throttle --------------------------------------------------------------

def test_second_check_is_throttled(state_dir):
    first = update_check.check_for_update(
        current_version="0.9.0", transport=_serving(_manifest()))
    assert first is not None
    second = update_check.check_for_update(
        current_version="0.9.0", transport=_serving(_manifest()))
    assert second is None, "a same-day restart must not re-check"


def test_throttle_expires(state_dir):
    update_check.mark_checked(
        dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=25))
    assert update_check.check_for_update(
        current_version="0.9.0", transport=_serving(_manifest())) is not None


def test_force_bypasses_the_throttle(state_dir):
    update_check.mark_checked()
    assert update_check.check_for_update(
        current_version="0.9.0", force=True,
        transport=_serving(_manifest())) is not None


def test_a_failed_check_still_counts_against_the_throttle(state_dir):
    update_check.check_for_update(
        current_version="0.9.0", transport=_failing())
    assert not update_check.due_for_check(), (
        "an offline machine must not retry on every single startup")


def test_disabled_stops_checking(state_dir):
    update_check.set_enabled(False)
    assert update_check.check_for_update(
        current_version="0.9.0", transport=_serving(_manifest())) is None
    assert update_check.check_for_update(
        current_version="0.9.0", force=True,
        transport=_serving(_manifest())) is not None, (
        "an explicit 'check now' still works while automatic checks are off")


# --- skip list -------------------------------------------------------------

def test_skipped_version_stays_silent(state_dir):
    update_check.skip_version("0.9.1")
    update_check.mark_checked(
        dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=25))
    assert update_check.check_for_update(
        current_version="0.9.0", transport=_serving(_manifest("0.9.1"))) is None


def test_skipping_one_version_does_not_swallow_the_next(state_dir):
    update_check.skip_version("0.9.1")
    update_check.mark_checked(
        dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=25))
    info = update_check.check_for_update(
        current_version="0.9.0", transport=_serving(_manifest("0.9.2")))
    assert info is not None and info.version == "0.9.2"


def test_corrupt_state_file_does_not_break_the_check(state_dir):
    with open(update_check.state_path(), "w", encoding="utf-8") as handle:
        handle.write("{ this is not json")
    assert update_check.check_for_update(
        current_version="0.9.0", transport=_serving(_manifest())) is not None


# --- what goes on the wire -------------------------------------------------

def test_manifest_url_carries_no_identifiers():
    url = update_check.manifest_url()
    assert url.startswith("https://")
    assert "?" not in url, "a query string is where identifiers leak in"
    assert update_check._channel() in url
