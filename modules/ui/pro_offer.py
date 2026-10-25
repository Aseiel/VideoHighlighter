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
  look for it: teaching one is free (``modules/teach``), and the banner says so
  first; looking by name with no training is Pro's.
* ``report_unmeasured`` -- the report they just opened lists things that were
  said and never measured, and the routes that could measure them need an
  engine this build does not ship. Which routes those are is read from
  :mod:`modules.report.detection_routes` itself, not listed here, so this cannot
  drift from what the advisor would recommend.
* ``object_unknown`` -- they typed an object the selected detector has no
  class for, so no run will ever find it. The banner says that first, because
  it is true whatever the edition (and it may only be a typo); teaching is the
  free fix, looking by name is Pro's.
* ``vr_region`` -- they ticked the side-by-side VR box. This edition analyses
  the whole frame either way; Pro analyses one eye, or the left, right or
  centre of the frame.
* ``lrv_proxy`` -- they added a recording whose camera left a low-resolution
  preview next to it (GoPro ``GL….LRV``, Insta360 ``LRV_…``). Pro analyses the
  preview and cuts from the original.

Never at startup, never on a timer, never in a modal dialog.

Asked, not offered
------------------
``face_category`` is never pushed: it is a menu entry marked "Pro" at the end
of the expression picker, and the banner appears only when the user clicks it.
Something the user asked about skips the cooldowns and the off switch (see
:func:`may_offer`) -- they are for interruptions, and this is an answer.

The same goes for :func:`footer_text`, the one line that is always there
while suggestions are on: it names the edition and links to the page. It sits
in a row that already exists, so it costs no space and never moves.

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
import html
import json
import os
import re
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Optional

from modules.i18n import t
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

MOMENTS = ("rule_unbuildable", "report_unmeasured", "object_unknown",
           "vr_region", "lrv_proxy", "face_category")


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


def may_offer(moment: str, now: Optional[_dt.datetime] = None,
              asked: bool = False) -> bool:
    """Whether ``moment`` may show an offer right now.

    Answers no in a Pro build, once switched off, and inside either cooldown.
    ``asked`` means the user clicked something marked Pro to find out what it
    is; only the edition is checked then. Does not record anything: call
    :func:`mark_shown` when the banner is actually on screen, so an offer that
    was decided but never displayed does not use up the cooldown.
    """
    if moment not in MOMENTS or not is_free_build():
        return False
    if asked:
        return True
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


# Every sentence below is a msgid in modules/i18n, translated here rather than
# by the Qt hooks: the banner receives it already formatted, and a filled-in
# sentence is never a key in any catalog.

def _trial_line() -> str:
    return t("Pro has a free {days}-day trial.").format(days=TRIAL_DAYS)


def _with_trial(sentence: str) -> str:
    return sentence + " " + _trial_line()


def footer_text() -> str:
    """The always-there line: which edition this is, and a link to Pro."""
    return t('Free edition · <a href="{url}">What\'s in Pro?</a>').format(
        url=PRO_URL)


def for_unbuildable_rule(now: Optional[_dt.datetime] = None) -> Optional[Offer]:
    """The offer after a rule proposal came back empty, or None."""
    if not may_offer("rule_unbuildable", now):
        return None
    return Offer("rule_unbuildable", _with_trial(t(
        "If what you asked about isn't one of this video's classes, no rule "
        "can add it — a rule only arranges what the detector found. You can "
        "teach a model to find it: right-click it in the player, then "
        "<i>Teach a model</i>. <b>VideoHighlighter Pro</b> can also look for "
        "it by name straight away, with no training.")))


def unknown_objects(typed: Iterable[str], known: Iterable[str]) -> list:
    """The typed object names the detector has no class for, in typed order.

    Exact, because the run matches exactly: "Person" finds nothing where the
    class is "person", so it is as unknown as a name that was never there.
    """
    known = set(known)
    out = []
    for name in typed:
        name = (name or "").strip()
        if name and name not in known and name not in out:
            out.append(name)
    return out


def for_unknown_objects(names: Iterable[str],
                        now: Optional[_dt.datetime] = None) -> Optional[Offer]:
    """The offer after the user typed objects the detector cannot find."""
    names = [n for n in names if n]
    if not names or not may_offer("object_unknown", now):
        return None
    return Offer("object_unknown", _with_trial(t(
        "The selected detector has no class called <b>{names}</b>, so a run "
        "will not find it. If it is not a typo, you can teach a model to find "
        "it: right-click it in the player, then <i>Teach a model</i>. "
        "<b>VideoHighlighter Pro</b> can also look for it by name straight "
        "away, with no training.").format(
            names=html.escape(", ".join(names)))))


def for_vr(now: Optional[_dt.datetime] = None) -> Optional[Offer]:
    """The offer after the side-by-side VR box was ticked."""
    if not may_offer("vr_region", now):
        return None
    return Offer("vr_region", _with_trial(t(
        "This edition analyses the whole frame of side-by-side footage. "
        "<b>VideoHighlighter Pro</b> analyses just one eye, or the left, "
        "right or centre of the frame, and can pick which on its own.")))


def for_face_category(now: Optional[_dt.datetime] = None) -> Optional[Offer]:
    """The answer to clicking the Pro entry in the expression picker."""
    if not may_offer("face_category", now, asked=True):
        return None
    return Offer("face_category", _with_trial(t(
        "These five expressions are what this edition reads on a face. "
        "<b>VideoHighlighter Pro</b> learns face categories of your own from "
        "a few example faces you pick, for anything the five do not cover.")))


# GoPro: GX/GH<chapter><file>.MP4 records GL<chapter><file>.LRV beside it.
# Insta360: VID_<stamp>_00_<n> records LRV_<stamp>_11_<n> beside it.
_GOPRO_MASTER = re.compile(r"^G[HX](\d{6})$", re.IGNORECASE)
_INSTA_MASTER = re.compile(r"^VID_(.+)_00_(\d+)$", re.IGNORECASE)


def lrv_companion(path: str) -> Optional[str]:
    """The file name of the camera's low-res preview beside ``path``, or None.

    Pairing by name only, the way the camera writes them; a ``.lrv`` added on
    its own counts as its own preview. Never raises.
    """
    name = os.path.basename(path or "")
    stem, ext = os.path.splitext(name)
    if ext.lower() == ".lrv":
        return name
    gopro = _GOPRO_MASTER.match(stem)
    insta = _INSTA_MASTER.match(stem)
    if gopro:
        wanted = {f"gl{gopro.group(1)}.lrv".lower()}
    elif insta:
        wanted = {f"lrv_{insta.group(1)}_11_{insta.group(2)}{e}".lower()
                  for e in (".lrv", ".mp4")}
    else:
        return None
    try:
        names = os.listdir(os.path.dirname(os.path.abspath(path)))
    except OSError:
        return None
    return next((n for n in sorted(names) if n.lower() in wanted), None)


def for_lrv(video_path: str, preview: str,
            now: Optional[_dt.datetime] = None) -> Optional[Offer]:
    """The offer after adding a recording that has a low-res preview."""
    if not preview or not may_offer("lrv_proxy", now):
        return None
    return Offer("lrv_proxy", _with_trial(t(
        "<b>{video}</b> has a low-resolution preview from the camera beside "
        "it ({preview}). <b>VideoHighlighter Pro</b> can analyse that preview "
        "instead, which is quicker on long high-resolution recordings, and "
        "still cut from the full-quality original.").format(
            video=html.escape(os.path.basename(video_path)),
            preview=html.escape(preview))))


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
    ways = "; ".join(r.name.lower() for r in missing)
    lead = (t("This report lists 1 thing that was said but not measured.")
            if n == 1 else
            t("This report lists {n} things that were said but not measured.")
            .format(n=n))
    return Offer("report_unmeasured", _with_trial(lead + " " + t(
        "<b>VideoHighlighter Pro</b> adds ways to measure them that this "
        "edition doesn't have: {ways}.").format(ways=ways)))
