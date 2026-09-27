"""Mention Pro's free trial, only where the free edition actually stops.

The free edition has no caps to hit: nothing counts runs, minutes or exports.
What it has are edges -- things a user reaches for that this build does not
ship. Telling them Pro exists is useful exactly there, and noise anywhere else,
so this module names the edges and decides whether a given one may say so now.

Where it appears
----------------
Only right after the user has run into one of these:

* ``rule_unbuildable`` -- they asked for a composition rule and none came back.
  A rule arranges classes the detector produced; it cannot add one. If what
  they asked about is not one of those classes, the fix is a detector that can
  look for it, and those are Pro's.
* ``report_unmeasured`` -- the report they just opened lists things that were
  said and never measured, and the routes that could measure them need an
  engine this build does not ship. Which routes those are is read from
  :mod:`modules.report.detection_routes` itself, not listed here, so this cannot
  drift from what the advisor would recommend.

Never at startup, never on a timer, never in a modal dialog.

Why not in the report
---------------------
The report, the findings and the advisor are identical in both editions
(README, "Pro edition"), and a report is something people send to other
people. An offer inside it would be an ad in a document the user handed to
someone else. So the report stays as it is and the offer is a banner in the
app, next to it.

How often
---------
Each edge speaks at most once every :data:`MOMENT_COOLDOWN_DAYS`, and any offer
at all at most once every :data:`GLOBAL_COOLDOWN_DAYS`, so a user who keeps
running into the same edge hears about it once and then gets on with their
work. One click switches it off for good; the About tab switches it back on.

What is sent
------------
Nothing. The state is a local file, and the only network access is the
browser opening :data:`PRO_URL` when the user clicks the button.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import dataclass
from typing import Callable, Mapping, Optional

from modules.system.app_paths import user_data_dir
from version import __edition__

# Where "Try Pro" goes. The site carries the trial and the download; keeping it
# the one place a user lands means a change of shop or trial terms is an edit
# to the site, not a release of this app.
PRO_URL = "https://aseiel.github.io/VideoHighlighter-site/"
TRIAL_DAYS = 14

MOMENT_COOLDOWN_DAYS = 30
GLOBAL_COOLDOWN_DAYS = 3
STATE_FILENAME = "pro_offer_state.json"

MOMENTS = ("rule_unbuildable", "report_unmeasured")


@dataclass
class Offer:
    moment: str
    text: str           # rich text for the banner, one or two sentences


# ---------------------------------------------------------------------------
# Local state
# ---------------------------------------------------------------------------

def state_path() -> str:
    return os.path.join(user_data_dir(), STATE_FILENAME)


def load_state() -> dict:
    try:
        with open(state_path(), "r", encoding="utf-8") as handle:
            state = json.load(handle)
        return state if isinstance(state, dict) else {}
    except (OSError, ValueError):
        return {}


def save_state(state: dict) -> None:
    try:
        with open(state_path(), "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2)
    except OSError as exc:
        print(f"pro_offer: could not save state: {exc}")


def is_enabled() -> bool:
    """False once the user has said not to suggest Pro."""
    return bool(load_state().get("enabled", True))


def set_enabled(enabled: bool) -> None:
    state = load_state()
    state["enabled"] = bool(enabled)
    save_state(state)


def mark_shown(moment: str, now: Optional[_dt.datetime] = None) -> None:
    state = load_state()
    stamp = (now or _now()).isoformat()
    shown = state.get("shown") if isinstance(state.get("shown"), dict) else {}
    shown[moment] = stamp
    state["shown"] = shown
    state["last_shown"] = stamp
    save_state(state)


def _now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def _parse(stamp) -> Optional[_dt.datetime]:
    if not stamp:
        return None
    try:
        when = _dt.datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    except ValueError:
        return None
    return when if when.tzinfo else when.replace(tzinfo=_dt.timezone.utc)


def _within(stamp, days: int, now: _dt.datetime) -> bool:
    when = _parse(stamp)
    return when is not None and (now - when) < _dt.timedelta(days=days)


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------

def is_free_build() -> bool:
    return (__edition__ or "").strip().lower() != "pro"


def may_offer(moment: str, now: Optional[_dt.datetime] = None) -> bool:
    """Whether ``moment`` may show an offer right now.

    Answers no in a Pro build, once switched off, and inside either cooldown.
    Does not record anything: call :func:`mark_shown` when the banner is
    actually on screen, so an offer that was decided but never displayed does
    not use up the cooldown.
    """
    if moment not in MOMENTS or not is_free_build():
        return False
    state = load_state()
    if not state.get("enabled", True):
        return False
    now = now or _now()
    shown = state.get("shown") if isinstance(state.get("shown"), dict) else {}
    if _within(shown.get(moment), MOMENT_COOLDOWN_DAYS, now):
        return False
    if _within(state.get("last_shown"), GLOBAL_COOLDOWN_DAYS, now):
        return False
    return True


def pro_only_routes(report: Mapping,
                    installed: Optional[Callable] = None) -> list:
    """Routes this run would be offered if the build had every engine.

    The difference between the advisor's catalogue with every engine present
    and with the ones this build has. In the free repo an engine that is
    missing is one that ships in Pro, which is what makes this the honest list
    of what Pro would add *for this report*, not a feature sheet.
    """
    from modules.report import detection_routes

    caps = detection_routes.capabilities(report)
    here = {r.id for r in detection_routes.available(caps, installed)}
    return [r for r in detection_routes.available(caps, lambda _m: True)
            if r.id not in here]


def _trial_line() -> str:
    return f"Pro has a free {TRIAL_DAYS}-day trial."


def for_unbuildable_rule(now: Optional[_dt.datetime] = None) -> Optional[Offer]:
    """The offer after a rule proposal came back empty, or None."""
    if not may_offer("rule_unbuildable", now):
        return None
    return Offer("rule_unbuildable",
                 "If what you asked about isn't one of this video's classes, "
                 "no rule can add it — a rule only arranges what the detector "
                 "found. <b>VideoHighlighter Pro</b> can look for it by name, "
                 "or learn it from a box you draw around one example. "
                 + _trial_line())


def for_report(report: Mapping,
               installed: Optional[Callable] = None,
               now: Optional[_dt.datetime] = None) -> Optional[Offer]:
    """The offer after opening a report, or None.

    Only when the report has unmeasured claims *and* some route to measuring
    them needs an engine this build lacks. A report whose gaps the free build
    can already close says nothing -- the advisor in it already says how.
    """
    if not may_offer("report_unmeasured", now):
        return None
    try:
        from modules.report import uncovered_claims
        claims = (uncovered_claims.ensure(dict(report)) or {}).get("claims") or []
        if not claims:
            return None
        missing = pro_only_routes(report, installed)
    except Exception as exc:                        # pragma: no cover - defensive
        print(f"pro_offer: report check skipped: {exc}")
        return None
    if not missing:
        return None
    n = len(claims)
    things = "thing that was" if n == 1 else "things that were"
    ways = "; ".join(r.name.lower() for r in missing)
    return Offer("report_unmeasured",
                 f"This report lists {n} {things} said but not measured. "
                 f"<b>VideoHighlighter Pro</b> adds ways to measure them that "
                 f"this edition doesn't have: {ways}. " + _trial_line())
