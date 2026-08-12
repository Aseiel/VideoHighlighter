"""
reel_plan.py — arrange clips into a short-form reel that tells a story.

Why this exists
===============
The highlight engine answers "which moments scored highest". A reel needs a
different question answered: "what happens first, what happens next, and how
long does each thing stay on screen". Those are not the same, and the gap
between them is why a montage of genuinely good moments can still be dull —
it opens on whatever happened first, holds every shot the same length, and
stops rather than ends.

So this module does three things the scorer cannot:

**Structure.** A reel is four sections, not a list. The hook (the opening
seconds) has to be the most striking thing you have, not the earliest. Context
establishes where and what. Escalation is the body and takes most of the
running time. The payoff ends it, and is held longer than anything else because
an ending that cuts away at the same rhythm as the middle does not read as an
ending.

**Pace.** Shot length is the main lever on how a reel feels, and the useful
range is much shorter than a highlight reel's: 1–2.5 s for energetic material
against the 6 s the engine happily produces. Within a reel the pace also moves
— faster through escalation, slower on the payoff — because uniform pacing is
what makes an edit feel machine-made.

**Order.** Everything after the hook stays in the order it was shot. Progression
is what makes a sequence read as a story rather than as a shuffle, and shooting
order is the only progression the footage actually carries.

What it does not do
===================
It does not judge which moment is most striking. It takes scores if the caller
has them and otherwise uses shooting order, because inventing a notion of
"interesting" here would duplicate — and quietly disagree with — the scoring
the engine already did.

Public API
==========
    PACES / STRUCTURE / LENGTHS
    Pace / Section / Shot
    plan_reel(sources, ...) -> Edl
    describe_plan(edl) -> str
"""

from __future__ import annotations

from dataclasses import dataclass, field

from modules.edl import Cut, Edl

# Shot-length bands. The numbers are the ones short-form editing actually uses;
# the names are what a person picking one would call the feel they want.
PACES: dict[str, "Pace"] = {}


@dataclass(frozen=True)
class Pace:
    key: str
    label: str
    min_shot: float
    max_shot: float

    @property
    def typical(self) -> float:
        return (self.min_shot + self.max_shot) / 2.0

    @property
    def cuts_per_minute(self) -> tuple[int, int]:
        """The same band expressed the way pacing is usually discussed."""
        return (int(60 / self.max_shot), int(60 / self.min_shot))


for _p in (
    Pace("calm", "Calm scenic or emotional", 3.0, 6.0),
    Pace("vlog", "Vlog or recap", 2.0, 4.0),
    Pace("energetic", "Energetic montage", 1.0, 2.5),
    Pace("intense", "Intense, comedy or music-heavy", 0.5, 1.5),
):
    PACES[_p.key] = _p

DEFAULT_PACE = "energetic"


@dataclass(frozen=True)
class Section:
    """One part of the story, as a share of the reel and a pace adjustment.

    ``share`` values are taken from where the sections actually fall in a
    24-second reel — hook 0–2 s, context 2–6 s, escalation 6–20 s, payoff the
    last 4 — so they scale to any length without the structure changing shape.

    ``prefers`` names the framing this section reads best in, as a soft
    preference rather than a filter: a shoot that is all wide shots still gets
    a payoff. See :mod:`modules.shot_type` for what the names mean.
    """
    name: str
    share: float
    pace_scale: float
    min_shots: int
    max_shots: int = 99
    prefers: tuple = ()

    def shot_length(self, pace: Pace) -> float:
        return max(0.2, pace.typical * self.pace_scale)


STRUCTURE: tuple[Section, ...] = (
    # One shot, slightly longer than the body's rhythm: the opening frame has
    # to be understood before the cutting starts, or the rest is noise. A face
    # stops a scroll better than a landscape does, so it is preferred here.
    Section("Hook", 0.08, 1.15, min_shots=1, max_shots=1,
            prefers=("close_subject",)),
    # Establishing: where this is and what is happening, which is what a wide
    # shot is for.
    Section("Context", 0.17, 0.95, min_shots=2, prefers=("wide",)),
    # No preference — the body is where alternating matters more than any
    # particular kind.
    Section("Escalation", 0.58, 0.85, min_shots=3),
    # Held, and a face if there is one: an ending on a person lands, and an
    # ending cut at the body's rhythm does not read as an ending at all.
    Section("Payoff", 0.17, 1.9, min_shots=1, max_shots=2,
            prefers=("close_subject",)),
)

# The three lengths worth testing, and what each is for.
LENGTHS: tuple[tuple[int, str], ...] = (
    (15, "One idea, one striking moment"),
    (24, "General-purpose storytelling"),
    (50, "Only when the story needs the setup"),
)

# Below this a shot is a flash rather than an image, whatever the pace says.
MIN_SHOT = 0.35


@dataclass
class Shot:
    """One planned shot: where it comes from and what job it does."""
    source: str
    start: float
    duration: float
    section: str
    text: str = ""
    kind: str = ""            # framing, so the next pick can alternate


@dataclass
class _Source:
    path: str
    duration: float
    score: float = 0.0
    taken: float = 0.0        # how much of it is already spoken for
    order: int = 0
    kind: str = ""            # framing, from modules.shot_type


def _sections_for(duration: float, pace: Pace, structure=STRUCTURE,
                  unit: float = 0.0) -> list[tuple[Section, float, int]]:
    """(section, shot length, shot count) for a reel of ``duration``.

    Shot length is decided first and quantised before the count is taken from
    it. Doing it the other way — count from the raw length, then round the
    length onto the grid — stretches every shot after the count is fixed, and a
    24-second request comes back as 33.

    A final pass adds or removes shots from the escalation until the total
    lands near the target. Escalation is the elastic one on purpose: the hook
    is a single shot by definition and the payoff is held, so neither can
    absorb the difference without changing what the reel is.
    """
    plan: list[tuple[Section, float, int]] = []
    for section in structure:
        seconds = max(MIN_SHOT, duration * section.share)
        length = section.shot_length(pace)
        # The payoff is held: at least twice the body's rhythm, and long enough
        # to register as an ending even on a short reel where its share is
        # only a couple of seconds.
        if section.pace_scale > 1.5:
            length = max(length, pace.typical * 1.8, 2.0)
        if unit:
            length = max(unit, round(length / unit) * unit)
        count = int(round(seconds / length)) or 1
        count = max(section.min_shots, min(section.max_shots, count))
        plan.append((section, length, count))

    def total() -> float:
        return sum(length * count for _, length, count in plan)

    # Index of the section that stretches. Falls back to the longest one when
    # a caller supplies a structure with no escalation.
    elastic = next((i for i, (s, _, _) in enumerate(plan)
                    if s.name == "Escalation"), None)
    if elastic is None:
        elastic = max(range(len(plan)), key=lambda i: plan[i][1] * plan[i][2])

    section, length, count = plan[elastic]
    while total() > duration * 1.06 and count > section.min_shots:
        count -= 1
        plan[elastic] = (section, length, count)
    while total() < duration * 0.94 and count < section.max_shots:
        count += 1
        plan[elastic] = (section, length, count)

    # Still long with the shot count at its floor: a slow pace against a short
    # target, where six-second shots simply do not fit in twenty-four seconds.
    # Shorten the body's shots instead of dropping more of them — losing the
    # progression matters more than holding the pace exactly.
    step = unit or 0.25
    while total() > duration * 1.06 and length - step >= MIN_SHOT:
        length -= step
        plan[elastic] = (section, length, count)
    return plan


def _pick(sources: list[_Source], want: float, *, allow_reuse: bool,
          prefers: tuple = (), avoid_kind: str = "") -> _Source | None:
    """The next source with room for a ``want``-second shot.

    Ranked rather than filtered, because every one of these is a preference
    that a small shoot has to be able to override:

    - **Not yet used**, so a reel spreads across the footage before taking a
      second slice of anything.
    - **The framing this section reads best in** (see :class:`Section`).
    - **Not the same framing as the shot before it.** This is the one that
      shows: six selfies followed by six landscapes is two edits stuck end to
      end, and alternating them is most of what makes a sequence feel
      arranged. It is a tie-break rather than a rule, so a shoot with only one
      kind still cuts.
    - **Shooting order**, to keep the progression forward.
    """
    usable = [s for s in sources if s.duration - s.taken >= want
              and (allow_reuse or s.taken == 0)]
    if not usable:
        if not allow_reuse:
            return None
        # Nothing has a full shot left; take from whatever has the most
        # remaining rather than dropping the shot and coming up short.
        remaining = [s for s in sources if s.duration - s.taken > MIN_SHOT]
        return max(remaining, key=lambda s: s.duration - s.taken) if remaining else None

    def rank(source: _Source) -> tuple:
        return (
            0 if source.taken == 0 else 1,
            0 if (prefers and source.kind in prefers) else 1,
            1 if (avoid_kind and source.kind == avoid_kind) else 0,
            source.taken,
            source.order,
        )

    return min(usable, key=rank)


def plan_reel(sources, *, duration: float = 24.0, pace: str = DEFAULT_PACE,
              structure=STRUCTURE, scores=None, title: str = "Reel",
              transition: str = "cut", transition_duration: float = 0.25,
              easing: str = "linear",
              texts=None, music: str = "", width: int = 0, height: int = 0,
              analysis=None, quantise: bool = True, shots_by_kind=None,
              classify: bool = True, log_fn=print) -> Edl:
    """Arrange ``sources`` into a reel and return it as a cut list.

    ``sources`` are clip paths — usually the engine's highlight files, which are
    already the good parts. ``scores`` optionally maps path to a number; the
    highest becomes the hook. ``texts`` maps a section name to a line of
    on-screen text ("Hook" is the one that matters, since most viewers start
    watching muted).

    ``analysis`` is a :class:`modules.music_analysis.MusicAnalysis`; with
    ``quantise`` the shot lengths are rounded to a musical unit chosen to suit
    them — the beat rather than the bar, because at 66 BPM a bar is 3.6 s and
    an energetic reel wants shots half that long.

    Everything after the hook keeps the order it was given, which is shooting
    order when the caller passes the engine's output unchanged.
    """
    from modules.video_probe import probe_video

    if pace not in PACES:
        raise ValueError(f"unknown pace {pace!r} — expected one of "
                         f"{', '.join(PACES)}")
    band = PACES[pace]
    scores = scores or {}
    texts = texts or {}

    pool: list[_Source] = []
    for i, path in enumerate(sources or []):
        try:
            length = float(probe_video(path)["duration"])
        except Exception:
            length = 0.0
        if length > MIN_SHOT:
            pool.append(_Source(path=path, duration=length,
                                score=float(scores.get(path, 0.0)), order=i))
    if not pool:
        raise ValueError("no usable clips to build a reel from")

    # Framing, so the sections can prefer the kind of shot they read best in
    # and the body can alternate. Optional in both directions: a caller that
    # already classified passes it in, and a caller that cannot afford the
    # measurement turns it off and gets the old order-only behaviour.
    kinds = dict(shots_by_kind or {})
    if classify and not kinds:
        try:
            from modules.shot_type import classify_all
            kinds = {p: s.kind for p, s in
                     classify_all([s.path for s in pool], log_fn=log_fn).items()}
        except Exception as exc:
            log_fn(f"⚠️ Could not classify framing ({exc}); "
                   f"picking on order alone")
    for source in pool:
        source.kind = str(kinds.get(source.path, "") or "")

    # The hook is the most striking thing available, not the earliest. With
    # scores that is what they say; without them, a close shot stops a scroll
    # better than a landscape, so framing decides — and failing both, the
    # first clip stands in rather than the module inventing a judgement the
    # engine already owns.
    hook_prefers = STRUCTURE[0].prefers if STRUCTURE else ()
    if scores:
        hook = max(pool, key=lambda s: (s.score, -s.order))
    else:
        preferred = [s for s in pool if s.kind in hook_prefers]
        hook = preferred[0] if preferred else pool[0]
    rest = [s for s in pool if s is not hook]

    unit = _musical_unit(analysis, band.typical) if (analysis and quantise) else 0.0
    if unit:
        log_fn(f"🎼 Snapping shots to {unit:.2f}s "
               f"({'beat' if unit < 2.0 else 'bar'})")

    shots: list[Shot] = []
    for section, target, count in _sections_for(duration, band, structure, unit):
        for n in range(count):
            is_hook = section.name == "Hook" and n == 0
            candidates = [hook] if is_hook else rest
            previous = shots[-1].kind if shots else ""
            picked = _pick(candidates, target, allow_reuse=not is_hook,
                           prefers=section.prefers, avoid_kind=previous)
            if picked is None:
                picked = _pick(pool, target, allow_reuse=True,
                               prefers=section.prefers, avoid_kind=previous)
            if picked is None:
                break
            available = picked.duration - picked.taken
            length = max(MIN_SHOT, min(target, available))
            shots.append(Shot(source=picked.path, start=picked.taken,
                              duration=length, section=section.name,
                              text=texts.get(section.name, "") if n == 0 else "",
                              kind=picked.kind))
            picked.taken += length

    if not shots:
        raise ValueError("could not place any shots — clips are too short")

    cuts: list[Cut] = []
    for i, shot in enumerate(shots):
        last = i == len(shots) - 1
        cuts.append(Cut(
            source=shot.source, start=shot.start,
            end=shot.start + shot.duration,
            transition="cut" if last else transition,
            transition_duration=0.0 if last else transition_duration,
            easing=easing,
            label=shot.section, text=shot.text))

    reel = Edl(title=title, cuts=cuts, music=music,
               width=int(width), height=int(height))
    log_fn(f"🎬 {title}: {len(cuts)} shots, {reel.duration:.0f}s, "
           f"{cuts_per_minute(reel):.0f} cuts/min ({band.label})")
    if reel.duration > duration * 1.12:
        # Not a rounding miss: the structure has a floor of one hook, two
        # context shots, three of escalation and a held payoff, and at a slow
        # pace those seven shots are simply longer than the target. Saying so
        # is more use than silently returning something half again as long.
        faster = _faster_than(pace)
        log_fn(f"⚠️ {duration:.0f}s is shorter than a {band.label.lower()} "
               f"story fits into ({reel.duration:.0f}s is the least it can be)"
               + (f" — try the {PACES[faster].label.lower()} pace" if faster else ""))
    return reel


def _faster_than(pace: str) -> str:
    """The next pace down, for suggesting a way out of an impossible target."""
    order = ["calm", "vlog", "energetic", "intense"]
    try:
        return order[order.index(pace) + 1]
    except (ValueError, IndexError):
        return ""


def minimum_duration(pace: str, structure=STRUCTURE, analysis=None) -> float:
    """The shortest reel this structure can make at ``pace``.

    Exposed so a UI can grey out or warn on a length before a render rather
    than after one.
    """
    band = PACES[pace]
    unit = _musical_unit(analysis, band.typical) if analysis else 0.0
    total = 0.0
    for section in structure:
        length = section.shot_length(band)
        if section.pace_scale > 1.5:
            length = max(length, band.typical * 1.8, 2.0)
        if unit:
            length = max(unit, round(length / unit) * unit)
        total += length * section.min_shots
    return total


def _musical_unit(analysis, target: float) -> float:
    """The grid short shots snap to — the beat.

    Bars are the right unit for a long film, where a shot spans several of
    them, and the wrong one here. At 66 BPM a bar is 3.6 s, so snapping to bars
    turns every pace into the same pace: a 1.75 s energetic shot and a 3 s vlog
    shot both round to 3.6 and the preset the user picked stops meaning
    anything. The beat is fine enough that each pace keeps its own length
    (2 beats, 3 beats, 5 beats) and every shot still lands on the music.
    """
    interval = float(getattr(analysis, "beat_interval", 0.0) or 0.0)
    if interval <= 0:
        return 0.0
    # A very slow track can have beats longer than a fast shot; halving keeps
    # the grid usable without leaving the music.
    while interval > max(MIN_SHOT, target) * 1.4 and interval > MIN_SHOT * 2:
        interval /= 2.0
    return interval


def cuts_per_minute(edl: Edl) -> float:
    """How often the picture changes, which is the number pacing is discussed
    in. Zero-length reels report 0 rather than dividing by it."""
    seconds = edl.duration
    return (len(edl.cuts) / seconds * 60.0) if seconds > 0 else 0.0


def describe_plan(edl: Edl) -> str:
    """A human summary of the plan, for the log and the UI.

    Grouped by section rather than listed shot by shot: a dozen near-identical
    lines is exactly the output nobody reads.
    """
    if not edl.cuts:
        return "empty reel"
    order: list[str] = []
    grouped: dict[str, list[Cut]] = {}
    for cut in edl.cuts:
        name = cut.label or "Shots"
        if name not in grouped:
            grouped[name] = []
            order.append(name)
        grouped[name].append(cut)

    lines = [f"{edl.duration:.0f}s, {len(edl.cuts)} shots, "
             f"{cuts_per_minute(edl):.0f} cuts/min"]
    at = 0.0
    for name in order:
        group = grouped[name]
        span = sum(c.duration for c in group)
        shortest = min(c.duration for c in group)
        longest = max(c.duration for c in group)
        length = (f"{shortest:.1f}s" if abs(longest - shortest) < 0.05
                  else f"{shortest:.1f}-{longest:.1f}s")
        lines.append(f"  {at:5.1f}s  {name:11} {len(group):2d} shot(s) @ {length}")
        at += span
    return "\n".join(lines)
