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

import math
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
    place: int = -1           # the spot it was shot from, so a reel can spread


@dataclass
class _Source:
    path: str
    duration: float
    score: float = 0.0
    cursor: float = 0.0       # nothing before this is still free
    used: float = 0.0         # how much has actually been taken
    order: int = 0
    kind: str = ""            # framing, from modules.shot_type
    windows: object = None    # modules.shot_window.ClipWindows, when measured
    place: int = -1           # which spot it was shot from, from shot_place
    look: object = None       # modules.shot_look.Look, when measured

    @property
    def free(self) -> float:
        """How much of the clip has not been spoken for yet."""
        return max(0.0, self.duration - self.cursor)

    @property
    def graded(self) -> bool:
        return bool(getattr(self.windows, "measured", False))

    def quality(self, want: float) -> float:
        """How good the best unused ``want`` seconds of this clip are, 0..1.

        Zero for an unmeasured clip, which makes every clip equal and leaves
        whatever ranking the caller does to fall through to its later keys.
        """
        if not self.graded:
            return 0.0
        return float(self.windows.best(want, after=self.cursor).score)

    def take(self, want: float, unit: float = 0.0) -> tuple[float, float]:
        """Reserve the next shot and return where it starts and how long it is.

        This is the whole point of the module's use of
        :mod:`modules.shot_window`. Before it, a shot began wherever the last
        one ended — which for the *first* slice of a clip means frame zero, and
        frame zero is where the camera is still being raised, swung round, or
        pulled out of a pocket. The clip is fine; its opening second is not.

        Forward-only on purpose. Windows are searched from the cursor rather
        than across the whole clip so two slices of one source can never
        overlap, which would show as the reel repeating itself.

        ``unit`` is the musical grid, when there is one. It matters here
        because skipping past an unsteady opening leaves the cursor at an
        arbitrary offset, so the *last* slice of a clip can end up with less
        room than a full shot needs. Truncating it to whatever is left puts
        that one shot off the beat and takes the whole reel with it; rounding
        down to a whole unit instead keeps the grid, at the cost of a fraction
        of a second the reel was already allowed to come in short by.
        """
        room = self.free
        if self.graded and room > 0:
            window = self.windows.best(want, after=self.cursor)
            start, length = window.start, window.duration
        else:
            start, length = self.cursor, min(want, room)

        left = max(0.0, self.duration - start)
        length = min(want, left)
        if unit and length < want - 1e-6:
            whole = math.floor(length / unit + 1e-6) * unit
            if whole >= MIN_SHOT:
                length = whole
        if length < MIN_SHOT:
            length = min(MIN_SHOT, left)

        # Rounded because the arithmetic above accumulates: a start that came
        # off the sample grid, subtracted from a probed duration, leaves a
        # length like 2.9999999999999996 rather than 3. Nothing downstream
        # cares about the difference except the beat grid, where a shot a
        # quadrillionth under the unit is no longer a whole number of beats
        # and the reel stops landing on the music. Six places is far below
        # the millisecond the cut list is written to.
        start = round(start, 6)
        length = round(min(length, max(0.0, self.duration - start)), 6)
        self.cursor = start + length
        self.used += length
        return start, length


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


def _footage_for(duration: float, pace: Pace, structure, unit: float,
                 transition: str, transition_duration: float) -> float:
    """How much *footage* a reel needs to run for ``duration`` on screen.

    Every transition overlaps the two shots it joins, so the reel is shorter
    than the sum of its parts by the total of every overlap. Planning straight
    to the requested number therefore delivers something visibly shorter — a
    24-second request with eleven 0.4-second joins came out at 19.8, and the
    cut list said so in its header while the planner carried on as though it
    had not.

    Solved by iteration rather than algebra because the shot count depends on
    the target and the overlap depends on the shot count. Three passes is
    plenty: each one moves the answer by the difference the last one left, and
    the sequence converges immediately for any sane transition length.
    """
    if transition == "cut" or transition_duration <= 0:
        return duration

    target = duration
    for _ in range(3):
        lengths: list[float] = []
        for section, length, count in _sections_for(target, pace, structure, unit):
            lengths.extend([length] * count)
        if len(lengths) < 2:
            return duration
        # The same clamp modules.transitions applies: a transition may not eat
        # more than a third of either shot it sits between.
        absorbed = sum(min(transition_duration, min(a, b) / 3.0)
                       for a, b in zip(lengths, lengths[1:]))
        moved = duration + absorbed
        if abs(moved - target) < 0.05:
            return moved
        target = moved
    return target


def _pick(sources: list[_Source], want: float, *, allow_reuse: bool,
          prefers: tuple = (), avoid_kind: str = "",
          place_use: dict = None, avoid_place: int = None,
          avoid_looks: list = None) -> _Source | None:
    """The next source with room for a ``want``-second shot.

    Ranked rather than filtered, because every one of these is a preference
    that a small shoot has to be able to override:

    - **From a spot the reel has not used yet.** The strongest of these, and
      the one that stops a montage reading as one shot with hiccups: standing
      still and pressing record twice is the commonest way a reel repeats
      itself, and no amount of picking better moments fixes it, because the
      moments are genuinely good and genuinely of the same thing. It is first
      because it is the coarsest — with more places than shots every candidate
      scores zero here and the preferences below do all the work, and it only
      speaks up to break a tie against showing the same view again.
    - **Not the same spot as the shot before it**, for when there are fewer
      places than shots and something has to be repeated. Repeating a view
      later is a montage; repeating it immediately is a mistake.
    - **Not the same picture as the shot before it.** Position cannot catch a
      landmark filmed from half a kilometre away — by position those are
      different places, which they are, and the viewer still sees the same
      lake twice. So the two are compared directly, and only against the
      immediately preceding shot: the comparison is reliable enough to say
      "these two are the same view" and not reliable enough to prune a whole
      reel on. See :mod:`modules.shot_look`.
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
    usable = [s for s in sources if s.free >= want
              and (allow_reuse or s.used == 0)]
    if not usable:
        if not allow_reuse:
            return None
        # Nothing has a full shot left; take from whatever has the most
        # remaining rather than dropping the shot and coming up short.
        remaining = [s for s in sources if s.free > MIN_SHOT]
        return max(remaining, key=lambda s: s.free) if remaining else None

    def looks_repeated(source: _Source) -> int:
        """2 when this repeats the shot before it, 1 when it repeats one
        earlier in the reel, 0 otherwise.

        Both are worth avoiding and they are not worth the same. Two shots of
        one view side by side read as a stutter; the same two a few seconds
        apart read as a montage that came back to something. Graded rather
        than forbidden because this is a preference like the rest — a shoot
        with nothing else to offer still cuts.
        """
        if source.look is None or not avoid_looks:
            return 0
        from modules.shot_look import same_view
        if avoid_looks[-1] is not None and same_view(source.look, avoid_looks[-1]):
            return 2
        return 1 if any(same_view(source.look, seen)
                        for seen in avoid_looks[:-1] if seen is not None) else 0

    def rank(source: _Source) -> tuple:
        return (
            (place_use or {}).get(source.place, 0),
            1 if (avoid_place is not None and source.place == avoid_place
                  and source.place >= 0) else 0,
            looks_repeated(source),
            0 if source.used == 0 else 1,
            0 if (prefers and source.kind in prefers) else 1,
            1 if (avoid_kind and source.kind == avoid_kind) else 0,
            source.used,
            source.order,
        )

    return min(usable, key=rank)


def plan_reel(sources, *, duration: float = 24.0, pace: str = DEFAULT_PACE,
              structure=STRUCTURE, scores=None, title: str = "Reel",
              transition: str = "cut", transition_duration: float = 0.25,
              easing: str = "linear", feather: float = 0.0,
              texts=None, music: str = "", width: int = 0, height: int = 0,
              analysis=None, quantise: bool = True, shots_by_kind=None,
              classify: bool = True, windows=None, settle: bool = True,
              places=None, track: str = "", spread: bool = True,
              log_fn=print) -> Edl:
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

    ``settle`` decides where inside each clip a shot begins. On (the default)
    every source is measured by :mod:`modules.shot_window` and each shot takes
    the best window it can still reach, rather than starting at frame zero —
    which is where the camera is very often still being raised or pulled out.
    Off, and shots begin at the first unused frame as they always did.
    ``windows`` accepts an already-measured ``{path: ClipWindows}`` so a caller
    that has paid for the measurement does not pay twice.

    ``spread`` keeps the reel off the same view twice. Clips are grouped by
    where and when they were shot (see :mod:`modules.shot_place`) and the reel
    takes one shot per spot before it reuses any — which is what stops four
    shots of one valley, filmed over ten minutes of standing still, all
    reaching the same edit. ``track`` optionally names a GPX file, used to
    place clips whose own metadata carries no GPS; ``places`` accepts an
    already-computed ``{path: place number}``.

    Everything after the hook keeps the order it was given, which is shooting
    order when the caller passes the engine's output unchanged. Settling moves
    the in-*point* of a shot, never its place in the running order: what makes
    a sequence read as a story is the progression, and reordering by picture
    quality would trade the story for a slightly cleaner frame.
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

    # Where inside each clip the camera has actually settled. Optional in both
    # directions for the same reasons framing is: a caller that already
    # measured passes it in, and a caller that cannot afford it turns it off
    # and every shot starts at the first unused frame as before.
    graded = dict(windows or {})
    if settle and not graded:
        try:
            from modules.shot_window import profile_all
            graded = profile_all([s.path for s in pool], log_fn=log_fn)
        except Exception as exc:
            log_fn(f"⚠️ Could not measure camera settling ({exc}); "
                   f"shots will start at the top of each clip")
    if settle:
        for source in pool:
            source.windows = graded.get(source.path)

    # Which spot each clip was shot from, so the reel can visit each once
    # before repeating any. Same bargain as the two measurements above: a
    # caller can supply it, and a shoot whose files carry neither a time nor a
    # position still cuts — every clip simply lands in a place of its own.
    numbered = dict(places or {})
    if spread and not numbered:
        try:
            from modules.shot_place import group, locate, read_track
            found = locate([s.path for s in pool],
                           track=read_track(track, log_fn=log_fn) if track else None,
                           log_fn=log_fn)
            numbered = group(found, log_fn=log_fn)
        except Exception as exc:
            log_fn(f"⚠️ Could not work out where the clips were shot ({exc}); "
                   f"the reel may show the same view twice")
    if spread:
        for source in pool:
            source.place = int(numbered.get(source.path, -1))

        # And what each one looks like, for the repeats position cannot see.
        try:
            from modules.shot_look import look_all
            appearance = look_all([s.path for s in pool], log_fn=log_fn)
            for source in pool:
                source.look = appearance.get(source.path)
        except Exception as exc:
            log_fn(f"⚠️ Could not compare how the clips look ({exc}); "
                   f"two shots of the same view may end up side by side")

    # The hook is the most striking thing available, not the earliest. With
    # scores that is what they say; without them, a close shot stops a scroll
    # better than a landscape, so framing decides — and among equally framed
    # clips, the one whose opening is actually watchable. Failing all of that,
    # the first clip stands in rather than the module inventing a judgement the
    # engine already owns.
    hook_prefers = STRUCTURE[0].prefers if STRUCTURE else ()
    if scores:
        hook = max(pool, key=lambda s: (s.score, -s.order))
    else:
        preferred = [s for s in pool if s.kind in hook_prefers] or pool
        # Rounded, so only a clip that is *clearly* steadier jumps the queue
        # and a field of near-identical clips still opens on the earliest one.
        hook = max(preferred,
                   key=lambda s: (round(s.quality(1.0), 2), -s.order))
    rest = [s for s in pool if s is not hook]

    unit = _musical_unit(analysis, band.typical) if (analysis and quantise) else 0.0
    if unit:
        log_fn(f"🎼 Snapping shots to {unit:.2f}s "
               f"({'beat' if unit < 2.0 else 'bar'})")

    # Plan enough footage that the reel runs for as long as was asked *after*
    # the transitions have taken their share of it.
    footage = _footage_for(duration, band, structure, unit,
                           transition, transition_duration)
    if footage > duration + 0.05:
        log_fn(f"⏱️ Planning {footage:.1f}s of footage so the reel runs "
               f"{duration:.0f}s once the transitions overlap")

    shots: list[Shot] = []
    # How many shots the reel has already taken from each spot. The first key
    # _pick ranks on, so every place is visited once before any is repeated.
    place_use: dict[int, int] = {}
    last_place: int | None = None
    used_looks: list = []

    for section, target, count in _sections_for(footage, band, structure, unit):
        for n in range(count):
            is_hook = section.name == "Hook" and n == 0
            candidates = [hook] if is_hook else rest
            previous = shots[-1].kind if shots else ""
            picked = _pick(candidates, target, allow_reuse=not is_hook,
                           prefers=section.prefers, avoid_kind=previous,
                           place_use=place_use, avoid_place=last_place,
                           avoid_looks=used_looks)
            if picked is None:
                picked = _pick(pool, target, allow_reuse=True,
                               prefers=section.prefers, avoid_kind=previous,
                               place_use=place_use, avoid_place=last_place,
                           avoid_looks=used_looks)
            if picked is None:
                break
            start, length = picked.take(target, unit)
            if length <= 0:
                continue
            shots.append(Shot(source=picked.path, start=start,
                              duration=length, section=section.name,
                              text=texts.get(section.name, "") if n == 0 else "",
                              kind=picked.kind, place=picked.place))
            if picked.place >= 0:
                place_use[picked.place] = place_use.get(picked.place, 0) + 1
                last_place = picked.place
            used_looks.append(picked.look)

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
            easing=easing, feather=float(feather),
            label=shot.section, text=shot.text))

    reel = Edl(title=title, cuts=cuts, music=music,
               width=int(width), height=int(height))
    log_fn(f"🎬 {title}: {len(cuts)} shots, {reel.duration:.0f}s, "
           f"{cuts_per_minute(reel):.0f} cuts/min ({band.label})")

    # Worth saying out loud: an in-point the user did not choose is the kind of
    # thing that looks like a bug until you know why it happened.
    moved = [c for c in cuts if c.start > 0.25]
    if moved:
        latest = max(c.start for c in moved)
        log_fn(f"✂️ {len(moved)} of {len(cuts)} shots start later than frame "
               f"zero (up to {latest:.1f}s in) — the camera was still being "
               f"placed at the top of those clips")

    # A repeated spot is the thing the viewer notices and the log never
    # mentioned, so say it either way: that the reel is all different views,
    # or that there was not enough footage for that and which ones doubled up.
    if spread and shots and any(s.place >= 0 for s in shots):
        visited = [s.place for s in shots if s.place >= 0]
        repeated = len(visited) - len(set(visited))
        if repeated:
            log_fn(f"📍 {len(set(visited))} different spot(s) across "
                   f"{len(visited)} shots — {repeated} had to be shown twice, "
                   f"which is what a shoot with fewer places than shots costs")
        else:
            log_fn(f"📍 Every shot is from a different spot")
    if reel.duration > duration * 1.12:
        # Not a rounding miss: the structure has a floor of one hook, two
        # context shots, three of escalation and a held payoff, and at a slow
        # pace those seven shots are simply longer than the target. Saying so
        # is more use than silently returning something half again as long.
        faster = _faster_than(pace)
        log_fn(f"⚠️ {duration:.0f}s is shorter than a {band.label.lower()} "
               f"story fits into ({reel.duration:.0f}s is the least it can be)"
               + (f" — try the {PACES[faster].label.lower()} pace" if faster else ""))
    elif reel.duration < duration * 0.9:
        # The other direction, and a different cause: the structure fitted, the
        # footage did not. Worth naming because the fix is more clips rather
        # than a different setting.
        log_fn(f"⚠️ Came up short at {reel.duration:.0f}s of the {duration:.0f}s "
               f"asked for — there is not enough usable footage to fill it")
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
