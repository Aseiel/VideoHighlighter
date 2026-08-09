"""
CompositionEngine — detects composite events from per-frame object detections.

A "composed event" fires when a configurable set of conditions holds
consistently over a short time window. Two kinds of condition exist:

*spatial* (``rules:``)
    Count how many boxes of a *source* class have their centre inside a box of
    a *region* class, and require the count to fall in a range.

*signal* (``signals:``)
    Compare a per-second measurement against a threshold, or a per-second label
    against a set of accepted values.

Signal conditions exist because the interesting combinations are not all
spatial. Audio level, vocal brightness and an expression reading are each a
value per second with no box to be inside anything, so a spatial-only engine
cannot express "this measurement is high *and* that label is showing" —
the conditions had to be evaluated in separate places and correlated by hand
afterwards, which is exactly the join this engine exists to perform.

Both kinds are ANDed: an event fires on a second where every one of its
conditions holds. A spec with only signal conditions is evaluated once per
second and needs no detections at all, so signal-only rules work on a video
that was never object-detected.

Configuration is loaded from a YAML file (see composition_rules.example.yaml);
the engine itself contains no domain-specific class names, signal names or
event names.
"""
from __future__ import annotations

import yaml
from collections import deque, defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class _Rule:
    """One condition: count how many *source* boxes have their centre inside
    a *region* box, and verify the count falls in [min_count, max_count]."""
    source_class: str
    region_class: str
    min_count: int = 1
    max_count: int = 999   # 999 = no upper limit


@dataclass
class _SignalCondition:
    """One condition on a per-second signal.

    Numeric signals are bounded with *min_value* / *max_value*; labelled ones
    are matched against *any_of*. A condition may carry both, so a signal that
    is sometimes a number and sometimes a label still has one way to be
    described -- but in practice a signal is one or the other and only the
    matching fields are set.

    A second with no value for the signal never satisfies the condition. That is
    the conservative reading: a missing measurement is not evidence, and the
    alternative would fire events wherever a detector had simply not run.
    """
    signal: str
    min_value: float | None = None
    max_value: float | None = None
    any_of: frozenset = field(default_factory=frozenset)

    # A second passes only if the condition also held across a run of at least
    # this many consecutive seconds. Without it every signal condition is a
    # question about one instant, and the things worth finding in a soundtrack
    # are mostly shapes: "loud for fifteen seconds" is a different claim from
    # "loud", and a spike of one second is what the first is meant to exclude.
    sustained_secs: int = 0

    # A second passes if the condition held anywhere within this many seconds
    # either side. Signals measured by different means do not land on the same
    # second: an expression reading exists only for the seconds a face was
    # readable, so requiring it to coincide exactly with an audio peak throws
    # away most real co-occurrences on a technicality about sampling.
    within_secs: int = 0

    def holds(self, value) -> bool:
        if value is None:
            return False
        if self.any_of:
            return str(value).strip().lower() in self.any_of
        try:
            number = float(value)
        except (TypeError, ValueError):
            return False
        if self.min_value is not None and number < self.min_value:
            return False
        if self.max_value is not None and number > self.max_value:
            return False
        return True


@dataclass
class _EventSpec:
    name: str
    label: str
    rules: list
    signals: list = field(default_factory=list)
    # Off means "do not evaluate", NOT "forget about". The name still has to be
    # reported by `event_names` so a previous pass's results get stripped —
    # otherwise disabling a rule leaves its last output on the timeline for
    # ever, which looks exactly like the rule still running.
    enabled: bool = True

    # An event shorter than this is discarded whole. Distinct from a signal
    # condition's `sustained_secs`, which asks how long one *measurement* held:
    # an event can satisfy every condition for a moment and still be too brief
    # to be the thing being looked for, and with several conditions ANDed the
    # overlap where all of them hold is routinely shorter than any of them.
    min_duration_secs: float = 0.0

    # Seconds to ignore at the start and at the end. Opening material is
    # titles, music beds and encoding artifacts, none of it content and all of
    # it shaped like signal; `modules/loudness_bursts.py` guards 120s for that
    # reason and measured 18 of 48 raw candidates falling inside it.
    #
    # They are separate because the two ends are not alike, which cost a real
    # detection to learn: a symmetric 120s guard threw away a hand-marked
    # episode that ended 94 seconds before the file did. What a recording builds
    # towards tends to sit at its end, so a closing guard discards the payload
    # while an opening one discards the titles. Set `ignore_end_secs` only when
    # the material genuinely has trailing junk.
    #
    # `ignore_edges_secs` still works and sets both, for rules written before
    # the two were told apart.
    ignore_start_secs: float = 0.0
    ignore_end_secs: float = 0.0

    window_secs: float = 0.75   # majority-vote smoothing window
    persist_secs: float = 0.5   # keep a ghost box this long after last seen


class CompositionEngine:
    """
    Applies composition rules to a stream of per-frame object detections.

    Usage::

        engine = CompositionEngine("composition_rules.yaml")
        sec_events, overlay_bboxes = engine.run(object_bboxes_cache)

    Each entry in *object_bboxes_cache* must be a dict::

        {
            'timestamp':   float,          # seconds from video start
            'objects':     list[str],
            'bboxes':      list[[x1n, y1n, wn, hn]],  # normalised top-left + size
            'confidences': list[float],
        }

    Returns
    -------
    sec_events : dict[int, list[str]]
        Composed event names keyed by integer second — ready to merge into
        ``object_detections`` in pipeline.py.
    overlay_bboxes : list[dict]
        Timestamp-tagged entries in the same format as *object_bboxes_cache*.
        A spatial event carries the union of the boxes it matched. A
        signal-only event has no location, so it is marked across the whole
        frame — see ``run``.
    """

    # How far apart two firing timestamps can be and still be one event.
    RUN_GAP_SECONDS = 1.5

    def __init__(self, rules_path: str | Path):
        self._specs = self._load(Path(rules_path))

    # ------------------------------------------------------------------ public

    @property
    def event_names(self) -> list:
        """Every event name this rule set can emit.

        Lets a caller re-running the engine over an existing result strip the
        previous pass first, so running it twice is idempotent rather than
        double-counting.
        """
        return [s.name for s in self._specs]

    @property
    def object_classes(self) -> list:
        """Every detection class the *enabled* spatial rules read.

        The counterpart of ``composition_signals.signal_names``: between them a
        caller can work out what a rule set needs measured before it can be
        applied, and go and get it. Disabled events are excluded — an unticked
        rule must not send anybody off to run a detection pass for boxes it
        will never look at. (``event_names`` includes them, deliberately, for
        the opposite reason: their previous output still has to be stripped.)
        """
        return sorted({cls
                       for spec in self._specs if spec.enabled
                       for rule in spec.rules
                       for cls in (rule.source_class, rule.region_class)
                       if cls})

    def run(self, bbox_cache: list[dict],
            signals: dict | None = None,
            duration: float | None = None) -> tuple[dict, list]:
        """Apply the rules. *signals* maps a signal name to per-second values.

        Each entry is either a sequence indexed by whole second or a mapping
        from second to value; ``_signal_value`` accepts both so a caller can
        pass a dense curve and a sparse reading side by side without converting
        either. Omitting *signals* entirely leaves the engine behaving exactly
        as it did before signal conditions existed.
        """
        if not self._specs:
            return {}, []

        signals = signals or {}
        active = [s for s in self._specs if s.enabled]
        spatial_specs = [s for s in active if s.rules]
        signal_only = [s for s in active if not s.rules and s.signals]
        signal_only_names = {s.name for s in signal_only}

        frames = sorted(bbox_cache, key=lambda e: float(e.get('timestamp', 0)))

        # Masks span the signals *and* the frames: a spatial rule gated by a
        # signal is evaluated at frame timestamps, which may run past the end of
        # a curve, and an out-of-range lookup has to be a definite "no value"
        # rather than an index error.
        length = self._signal_span(signals)
        if frames:
            length = max(length, int(float(frames[-1].get('timestamp', 0))) + 1)
        masks = self._signal_masks(active, signals, length)

        # Per-spec state: rolling window + ghost tracker
        windows: dict[str, deque] = {s.name: deque() for s in active}
        # ghosts[spec_name][class] = [{'ts': float, 'box': list, 'conf': float}]
        ghosts: dict[str, dict] = {s.name: defaultdict(list) for s in active}

        raw_events: dict[float, set] = defaultdict(set)
        raw_boxes: dict[float, dict] = defaultdict(dict)   # ts → {event_name: [boxes]}

        for entry in frames:
            ts = float(entry.get('timestamp', 0))
            dets = self._parse_frame(entry)

            for spec in spatial_specs:
                # --- expire old ghosts ---
                for cls in list(ghosts[spec.name].keys()):
                    ghosts[spec.name][cls] = [
                        g for g in ghosts[spec.name][cls]
                        if ts - g['ts'] <= spec.persist_secs
                    ]

                # --- refresh / add live detections into ghost tracker ---
                for cls, boxes in dets.items():
                    for det in boxes:
                        existing = ghosts[spec.name][cls]
                        matched = False
                        for g in existing:
                            if self._iou(g['box'], det['box']) > 0.3:
                                g['ts'] = ts
                                g['box'] = det['box']
                                g['conf'] = det['conf']
                                matched = True
                                break
                        if not matched:
                            existing.append({'ts': ts, 'box': det['box'], 'conf': det['conf']})

                # --- build effective detections = live ghosts ---
                effective: dict = defaultdict(list)
                for cls, gs in ghosts[spec.name].items():
                    effective[cls].extend(gs)

                # --- evaluate rules ---
                fired, matched_boxes = self._evaluate(spec, effective)

                # --- AND in the signal conditions ---
                # Tested per frame rather than per second so a spec's spatial
                # and signal halves are answering about the same instant. The
                # signal itself only changes once a second; what varies inside
                # one is whether the boxes were there at the same time.
                if fired and spec.signals and not self._signals_hold(
                        spec, ts, masks):
                    fired, matched_boxes = False, []

                # --- rolling majority-vote window ---
                win = windows[spec.name]
                win.append((ts, fired, matched_boxes))
                while win and ts - win[0][0] > spec.window_secs:
                    win.popleft()

                if win and sum(1 for _, f, _ in win if f) / len(win) >= 0.5:
                    raw_events[ts].add(spec.name)
                    # Use matched boxes from the most recent fired frame
                    last_boxes = next(
                        (b for _, f, b in reversed(list(win)) if f), []
                    )
                    raw_boxes[ts][spec.name] = last_boxes

        # --- specs made only of signal conditions ---
        # Evaluated straight over whole seconds: they do not depend on frames,
        # which may be sparse or absent entirely, and the majority-vote window
        # would smooth a curve that is already one value per second.
        for sec in range(length if signal_only else 0):
            for spec in signal_only:
                if self._signals_hold(spec, float(sec), masks):
                    raw_events[float(sec)].add(spec.name)

        # --- drop what falls in the guarded ends ---
        # Inferred from the signals when the caller did not say, because a rule
        # asking to ignore the last two minutes cannot be honoured without
        # knowing where the end is, and silently ignoring only the opening would
        # be a worse answer than none.
        span = float(duration) if duration else float(length or 0)
        for spec in active:
            if span <= 0 or (spec.ignore_start_secs <= 0
                             and spec.ignore_end_secs <= 0):
                continue
            lo = spec.ignore_start_secs
            hi = span - spec.ignore_end_secs if spec.ignore_end_secs > 0 else span
            for t in list(raw_events):
                if (t < lo or t > hi) and spec.name in raw_events[t]:
                    raw_events[t].discard(spec.name)
                    raw_boxes.get(t, {}).pop(spec.name, None)

        # --- discard events too brief to be what was asked for ---
        for spec in active:
            if spec.min_duration_secs <= 0:
                continue
            times = sorted(ts for ts, names in raw_events.items()
                           if spec.name in names)
            if not times:
                continue
            runs, run = [], [times[0]]
            for t in times[1:]:
                # A gap wider than this starts a new event. Signal rules land on
                # whole seconds and spatial ones on frame timestamps, so the
                # tolerance has to clear a one-second step without bridging a
                # real silence between two separate events.
                if t - run[-1] <= self.RUN_GAP_SECONDS:
                    run.append(t)
                else:
                    runs.append(run)
                    run = [t]
            runs.append(run)
            for run in runs:
                # Measured the way the timeline draws it: a bar runs from the
                # first second to the last plus one, so a lone second is 1s long
                # rather than 0 and `min_duration_secs: 1` keeps it.
                if (run[-1] - run[0] + 1.0) >= spec.min_duration_secs:
                    continue
                for t in run:
                    raw_events[t].discard(spec.name)
                    raw_boxes.get(t, {}).pop(spec.name, None)
        raw_events = {t: names for t, names in raw_events.items() if names}

        # --- aggregate to per-second ---
        sec_events: dict = defaultdict(list)
        overlay_bboxes: list = []

        for ts in sorted(raw_events):
            sec = int(ts)
            for name in sorted(raw_events[ts]):
                if name not in sec_events[sec]:
                    sec_events[sec].append(name)
                matched = raw_boxes[ts].get(name, [])
                union = self._union_box([m['box'] for m in matched])
                if union is None and name in signal_only_names:
                    # A signal-only event has no location -- it is a statement
                    # about the whole moment, not about a thing in the frame.
                    # Without a box nothing draws at all and the event is
                    # invisible during playback, so it is marked across the
                    # frame: a border says "this moment", where a box somewhere
                    # in particular would claim a place it does not have.
                    union = [0.0, 0.0, 1.0, 1.0]
                    # No confidence: the rule held or it did not. Writing 1.0
                    # would put a boolean into the field detectors use for
                    # certainty -- the mistake this block already warns about
                    # below -- and 0.0 would read as "certainly not".
                    overlay_bboxes.append({
                        'timestamp': ts,
                        'objects': [name],
                        'bboxes': [union],
                        'confidences': [],
                    })
                    continue
                # A rule needs *every* detection it matched, so the event is
                # only as sure as its weakest one. This used to emit 1.0, which
                # put a boolean into the same field every detector writes its
                # certainty into — and readers of that field, quite reasonably,
                # took it to mean the detector was certain. A rule firing on two
                # barely-there detections is not the same evidence as one firing
                # on two solid ones, and now it does not claim to be.
                confidence = min((m['conf'] for m in matched), default=0.0)
                overlay_bboxes.append({
                    'timestamp': ts,
                    'objects': [name],
                    'bboxes': [union] if union else [],
                    'confidences': [round(float(confidence), 3)] if union else [],
                })

        return dict(sec_events), overlay_bboxes

    # ----------------------------------------------------------------- private

    @staticmethod
    def _signal_value(source, sec: int):
        """One signal's value at a whole second, or ``None`` if it has none.

        Accepts a sequence indexed by second or a mapping keyed by second, so
        callers can hand over a dense curve and a sparse per-second reading
        without converting either into the other's shape.
        """
        if source is None:
            return None
        if isinstance(source, dict):
            # A mapping may be keyed by int or by str depending on whether it
            # has been through JSON; a cached run has, a fresh one has not.
            if sec in source:
                return source[sec]
            return source.get(str(sec))
        try:
            if 0 <= sec < len(source):
                return source[sec]
        except TypeError:
            return None
        return None

    def _condition_mask(self, condition: _SignalCondition,
                        signals: dict, length: int) -> list:
        """Per-second truth for one condition, with its shape modifiers applied.

        Built once per run rather than tested per frame, because ``sustained``
        is not a property of a second at all -- it is a property of the run the
        second belongs to, and cannot be answered by looking at one value.
        """
        source = signals.get(condition.signal)
        base = [condition.holds(self._signal_value(source, s))
                for s in range(length)]

        if condition.sustained_secs > 1:
            need = condition.sustained_secs
            held = [False] * length
            start = None
            for i in range(length + 1):
                if i < length and base[i]:
                    if start is None:
                        start = i
                elif start is not None:
                    if i - start >= need:
                        # The whole run passes, not just its tail: an event that
                        # only began once the requirement was met would start
                        # `need` seconds after the thing it is reporting.
                        for j in range(start, i):
                            held[j] = True
                    start = None
            base = held

        if condition.within_secs > 0:
            reach = condition.within_secs
            near = [False] * length
            for i, on in enumerate(base):
                if on:
                    for j in range(max(0, i - reach),
                                   min(length, i + reach + 1)):
                        near[j] = True
            base = near

        return base

    def _signal_masks(self, specs: list, signals: dict, length: int) -> dict:
        """One mask per condition, keyed by identity."""
        return {id(c): self._condition_mask(c, signals, length)
                for spec in specs for c in spec.signals}

    @staticmethod
    def _signals_hold(spec: _EventSpec, ts: float, masks: dict) -> bool:
        """True when every one of *spec*'s signal conditions holds at *ts*."""
        sec = int(ts)
        for condition in spec.signals:
            mask = masks.get(id(condition)) or []
            if not (0 <= sec < len(mask)) or not mask[sec]:
                return False
        return True

    @staticmethod
    def _signal_span(signals: dict) -> int:
        """How many whole seconds the supplied signals cover.

        The longest of them, because a rule may combine a dense audio curve with
        a sparse per-second reading and the event can occur anywhere either one
        reaches.
        """
        longest = 0
        for source in (signals or {}).values():
            if source is None:
                continue
            if isinstance(source, dict):
                keys = [int(k) for k in source.keys()
                        if str(k).lstrip('-').isdigit()]
                longest = max(longest, (max(keys) + 1) if keys else 0)
            else:
                try:
                    longest = max(longest, len(source))
                except TypeError:
                    continue
        return longest

    @staticmethod
    def _load(path: Path) -> list:
        if not path.exists():
            return []
        with open(path, encoding='utf-8') as f:
            raw = yaml.safe_load(f) or {}
        specs = []
        for ev in raw.get('events', []):
            rules = [
                _Rule(
                    source_class=r['source'],
                    region_class=r['region'],
                    min_count=int(r.get('min_count', 1)),
                    max_count=int(r.get('max_count', 999)),
                )
                for r in ev.get('rules', [])
            ]
            signals = [
                _SignalCondition(
                    signal=str(s['signal']),
                    min_value=(None if s.get('min') is None
                               else float(s['min'])),
                    max_value=(None if s.get('max') is None
                               else float(s['max'])),
                    # `equals` is the single-value spelling of `any_of`; both
                    # land in the same set so the matching code has one path.
                    any_of=frozenset(
                        str(v).strip().lower()
                        for v in ([s['equals']] if s.get('equals') is not None
                                  else (s.get('any_of') or []))
                    ),
                    sustained_secs=int(s.get('sustained_secs', 0) or 0),
                    within_secs=int(s.get('within_secs', 0) or 0),
                )
                for s in ev.get('signals', [])
                if s.get('signal')
            ]
            specs.append(_EventSpec(
                enabled=bool(ev.get('enabled', True)),
                min_duration_secs=float(ev.get('min_duration_secs', 0) or 0),
                ignore_start_secs=float(
                    ev.get('ignore_start_secs',
                           ev.get('ignore_edges_secs', 0)) or 0),
                ignore_end_secs=float(
                    ev.get('ignore_end_secs',
                           ev.get('ignore_edges_secs', 0)) or 0),
                name=ev['name'],
                label=ev.get('label', ev['name']),
                rules=rules,
                signals=signals,
                window_secs=float(ev.get('window_secs', 0.75)),
                persist_secs=float(ev.get('persist_secs', 0.5)),
            ))
        return specs

    @staticmethod
    def _parse_frame(entry: dict) -> dict:
        result: dict = defaultdict(list)
        objs = entry.get('objects', [])
        boxes = entry.get('bboxes', [])
        confs = entry.get('confidences', [])
        for i, cls in enumerate(objs):
            box = boxes[i] if i < len(boxes) else [0.0, 0.0, 0.0, 0.0]
            conf = confs[i] if i < len(confs) else 1.0
            result[cls].append({'box': list(box), 'conf': float(conf)})
        return result

    @staticmethod
    def _centre_inside(source_box: list, region_box: list) -> bool:
        """True if the centre of *source_box* falls inside *region_box*.
        Both are [x1n, y1n, wn, hn] normalised."""
        sx1, sy1, sw, sh = source_box
        cx, cy = sx1 + sw / 2, sy1 + sh / 2
        rx1, ry1, rw, rh = region_box
        return rx1 <= cx <= rx1 + rw and ry1 <= cy <= ry1 + rh

    @staticmethod
    def _iou(a: list, b: list) -> float:
        ax1, ay1, aw, ah = a
        bx1, by1, bw, bh = b
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    def _evaluate(self, spec: _EventSpec, effective: dict) -> tuple:
        """Returns (fired: bool, matched: list[{'box', 'conf'}]).

        The detections are returned whole rather than reduced to geometry: an
        event's confidence is derived from theirs, and a box alone cannot say
        how sure anything was.


        Source instances are consumed across rules — a source object that
        satisfies rule 1 is removed from the pool before rule 2 is checked,
        so two rules each requiring 1 source genuinely need 2 distinct ones.
        """
        all_matched = []
        # Mutable per-class pools so each source instance can only be claimed once
        available: dict = {cls: list(dets) for cls, dets in effective.items()}

        for rule in spec.rules:
            sources = available.get(rule.source_class, [])
            regions = effective.get(rule.region_class, [])
            if not regions:
                if rule.min_count > 0:
                    return False, []
                continue
            # Claim source instances greedily; each source counts at most once
            claimed, claimed_idx = [], []
            for i, src in enumerate(sources):
                if any(self._centre_inside(src['box'], rgn['box']) for rgn in regions):
                    claimed.append(src)
                    claimed_idx.append(i)
            count = len(claimed)
            if not (rule.min_count <= count <= rule.max_count):
                return False, []
            # Remove claimed sources so later rules can't reuse them
            available[rule.source_class] = [
                s for i, s in enumerate(sources) if i not in claimed_idx
            ]
            # Carry the detections themselves, not just their geometry: the
            # caller needs their confidences to say how sure the event is.
            all_matched.extend(claimed)
            all_matched.extend(regions)
        return True, all_matched

    @staticmethod
    def _union_box(boxes: list) -> list | None:
        """Smallest axis-aligned box covering all input [x1n,y1n,wn,hn] boxes."""
        if not boxes:
            return None
        x1 = min(b[0] for b in boxes)
        y1 = min(b[1] for b in boxes)
        x2 = max(b[0] + b[2] for b in boxes)
        y2 = max(b[1] + b[3] for b in boxes)
        return [x1, y1, x2 - x1, y2 - y1]
