"""
CompositionEngine — detects composite events from per-frame object detections.

A "composed event" fires when a configurable set of spatial relations between
detected object classes holds consistently over a short time window.

Configuration is loaded from a YAML file (see composition_rules.example.yaml);
the engine itself contains no domain-specific class names or event names.
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
class _EventSpec:
    name: str
    label: str
    rules: list
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
        Timestamp-tagged entries in the same format as *object_bboxes_cache*,
        with empty bbox lists (the events have no extra box to draw).
    """

    def __init__(self, rules_path: str | Path):
        self._specs = self._load(Path(rules_path))

    # ------------------------------------------------------------------ public

    def run(self, bbox_cache: list[dict]) -> tuple[dict, list]:
        if not self._specs:
            return {}, []

        frames = sorted(bbox_cache, key=lambda e: float(e.get('timestamp', 0)))

        # Per-spec state: rolling window + ghost tracker
        windows: dict[str, deque] = {s.name: deque() for s in self._specs}
        # ghosts[spec_name][class] = [{'ts': float, 'box': list, 'conf': float}]
        ghosts: dict[str, dict] = {s.name: defaultdict(list) for s in self._specs}

        raw_events: dict[float, set] = defaultdict(set)
        raw_boxes: dict[float, dict] = defaultdict(dict)   # ts → {event_name: [boxes]}

        for entry in frames:
            ts = float(entry.get('timestamp', 0))
            dets = self._parse_frame(entry)

            for spec in self._specs:
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

        # --- aggregate to per-second ---
        sec_events: dict = defaultdict(list)
        overlay_bboxes: list = []

        for ts in sorted(raw_events):
            sec = int(ts)
            for name in sorted(raw_events[ts]):
                if name not in sec_events[sec]:
                    sec_events[sec].append(name)
                union = self._union_box(raw_boxes[ts].get(name, []))
                overlay_bboxes.append({
                    'timestamp': ts,
                    'objects': [name],
                    'bboxes': [union] if union else [],
                    'confidences': [1.0] if union else [],
                })

        return dict(sec_events), overlay_bboxes

    # ----------------------------------------------------------------- private

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
            specs.append(_EventSpec(
                name=ev['name'],
                label=ev.get('label', ev['name']),
                rules=rules,
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
        """Returns (fired: bool, matched_boxes: list[box]).

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
            all_matched.extend(s['box'] for s in claimed)
            all_matched.extend(r['box'] for r in regions)
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
