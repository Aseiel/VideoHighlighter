"""
auto_pipeline.py — card in, film out, as one resumable operation.

Why this exists
===============
Every piece of the journey from a camera card to a finished film already exists
somewhere in this repo: :mod:`modules.gopro_ingest` copies footage off a card,
``pipeline.run_highlighter`` scores a video and cuts the good parts,
:mod:`modules.combine_videos` joins clips into a reel, and
:mod:`modules.music_track` lays music over the result. What did not exist is
anything that runs them *in order* and survives being interrupted.

That matters more than it sounds. The full sequence over a 4 GB card is tens of
minutes of mostly-unattended work, and the expensive middle (detection over
every frame of every clip) is the part most likely to be interrupted — a
cancel, a crash in a native detector, a laptop lid. Re-running from the top
each time makes the tool unusable for its actual purpose. So every stage here
records what it produced, and a re-run skips whatever is already on disk.

Design
======
The pipeline is a list of named stages with an explicit state file
(``job.json``) next to the output. A stage is skipped when its recorded outputs
still exist, which makes ``resume`` the default rather than a special mode.
State is written after *every* stage transition, because the failure this
protects against is precisely the one where the process does not get to run its
cleanup.

Stages are deliberately coarse — ingest, music, highlight, combine, score —
because that is the granularity at which a user thinks about the work and at
which re-running is cheap enough to be honest. Finer resume points inside
detection belong to the engine's own caches, which already exist.

Heavy imports (the engine, which drags in torch) are deferred into the stage
that needs them, so this module stays importable — and testable — in an
environment with none of that installed. ``highlight_runner`` exists for the
same reason: tests inject a stub instead of running real detection.

Nothing here re-implements engine behaviour. If a stage looks like it is making
an editorial decision, it is passing the decision to the module that owns it.

Public API
==========
    run_auto_pipeline(...) -> JobResult
    load_job(path) -> JobState
    JobState / Stage / JobResult
"""

from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import dataclass, field, asdict

# Stage names are part of the on-disk state format: renaming one invalidates
# resume for jobs already in flight, so they are constants rather than literals
# scattered through the code.
STAGE_INGEST = "ingest"
STAGE_MUSIC = "music"
STAGE_HIGHLIGHT = "highlight"
STAGE_COMBINE = "combine"
STAGE_MUSIC_MIX = "music_mix"

STAGE_ORDER = (STAGE_INGEST, STAGE_MUSIC, STAGE_HIGHLIGHT,
               STAGE_COMBINE, STAGE_MUSIC_MIX)

PENDING = "pending"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
SKIPPED = "skipped"

JOB_FILE = "job.json"
STATE_VERSION = 1


class PipelineCancelled(RuntimeError):
    """Raised at a stage boundary when cancel_check() says stop.

    Cancellation is checked between stages and inside the long ones, never
    mid-ffmpeg: a half-written output file that looks finished is worse than
    doing the work again.
    """


@dataclass
class Stage:
    name: str
    status: str = PENDING
    detail: str = ""
    error: str = ""
    seconds: float = 0.0
    outputs: list[str] = field(default_factory=list)
    # Fingerprint of the settings this stage's output was produced under.
    # Empty means "no settings worth tracking", which is most stages.
    key: str = ""

    def satisfied_by(self, key: str = "") -> bool:
        """True when this stage's recorded work is still on disk *and* was made
        under the settings being asked for now.

        Two ways a resume must redo a stage. The obvious one is that its output
        was deleted — marching past that fails later with a confusing
        missing-file error from a stage that is not the problem. The other is
        that the request changed: asking for crossfades at 1080p and getting
        back the hard-cut 5.3K reel from the previous run, because a file of
        the right name happened to exist, is the kind of thing you only notice
        after watching it.
        """
        if self.status not in (DONE, SKIPPED):
            return False
        if key and self.key != key:
            return False
        return all(os.path.exists(p) for p in self.outputs)

    @property
    def is_satisfied(self) -> bool:
        return self.satisfied_by()


@dataclass
class JobState:
    job_id: str
    root: str
    version: int = STATE_VERSION
    created: str = ""
    stages: dict[str, Stage] = field(default_factory=dict)
    clips: list[str] = field(default_factory=list)
    highlights: list[str] = field(default_factory=list)
    reel: str = ""
    final: str = ""
    music: str = ""
    music_analysis: str = ""
    script: str = ""
    edl: str = ""
    errors: list[str] = field(default_factory=list)

    def stage(self, name: str) -> Stage:
        return self.stages.setdefault(name, Stage(name=name))

    def to_dict(self) -> dict:
        data = asdict(self)
        data["stages"] = {k: asdict(v) for k, v in self.stages.items()}
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "JobState":
        stages = {k: Stage(**v) for k, v in (data.get("stages") or {}).items()}
        known = {f for f in cls.__dataclass_fields__ if f != "stages"}
        return cls(stages=stages,
                   **{k: v for k, v in data.items() if k in known})


@dataclass
class JobResult:
    state: JobState
    output: str = ""          # the film, whatever the last successful stage produced
    seconds: float = 0.0
    ok: bool = True

    @property
    def failed_stages(self) -> list[str]:
        return [n for n, s in self.state.stages.items() if s.status == FAILED]


def job_path(root: str) -> str:
    return os.path.join(root, JOB_FILE)


def save_job(state: JobState, path: str = "") -> str:
    """Persist job state. Written via a temp file + replace so an interruption
    during the write cannot leave a truncated state file — which would strand
    the very job this exists to make resumable."""
    path = path or job_path(state.root)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state.to_dict(), fh, indent=2)
    os.replace(tmp, path)
    return path


def load_job(path: str) -> JobState:
    """Load job state written by :func:`save_job`.

    Raises on an unreadable or future-versioned file: silently starting from
    scratch would quietly redo an hour of detection, and silently trusting an
    unknown schema would resume into undefined behaviour.
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if data.get("version") != STATE_VERSION:
        raise ValueError(
            f"unsupported job state version: {data.get('version')!r} "
            f"(this build understands {STATE_VERSION})")
    return JobState.from_dict(data)


def _check_cancel(cancel_check) -> None:
    if cancel_check is not None and cancel_check():
        raise PipelineCancelled("cancelled")


def _default_highlight_runner(paths, gui_config, log_fn, progress_fn, cancel_flag):
    """Call the real engine. Deferred import: ``pipeline`` pulls in torch, cv2
    and the detectors, which must not be a condition of importing this module."""
    from pipeline import run_highlighter
    return run_highlighter(paths, gui_config=gui_config, log_fn=log_fn,
                           progress_fn=progress_fn, cancel_flag=cancel_flag)


def run_auto_pipeline(
    *,
    dest_root: str,
    card=None,
    source_paths: list[str] | None = None,
    folder_name: str = "",
    script_path: str = "",
    music_path: str = "",
    config: dict | None = None,
    output_name: str = "film.mp4",
    music_mode: str = "replace",
    music_volume: float = 0.8,
    transition: str = "cut",
    transition_duration: float = 0.5,
    transition_bars: float = 0.0,
    quantise: str = "",
    width: int = 0,
    height: int = 0,
    fps: int = 0,
    crf: int = 20,
    resume: bool = True,
    verify: str = "size",
    job_id: str = "",
    highlight_runner=None,
    log_fn=print,
    progress_fn=None,
    stage_fn=None,
    cancel_check=None,
) -> JobResult:
    """Run card-to-film end to end, resuming whatever a previous run finished.

    Exactly one source is required: ``card`` (a
    :class:`modules.gopro_ingest.GoProCard` to copy from) or ``source_paths``
    (footage already on disk). ``script_path`` and ``music_path`` are optional;
    without them this degrades to "ingest, highlight, combine", which is still
    the useful default.

    ``progress_fn(fraction, detail)`` reports overall progress in 0..1.
    ``stage_fn(stage_name, status, detail)`` fires on every stage transition, so
    a UI can render the pipeline rather than a single opaque bar.
    ``highlight_runner`` overrides the engine call for testing.

    Returns a :class:`JobResult`; ``ok`` is False when any stage failed. A
    failed stage does not necessarily mean no output — a music mux that fails
    still leaves a perfectly good silent reel, and that is reported as the
    output rather than thrown away.
    """
    started = time.time()
    if card is None and not source_paths:
        raise ValueError("run_auto_pipeline needs either card= or source_paths=")

    os.makedirs(dest_root, exist_ok=True)
    state_path = job_path(dest_root)

    state = None
    if resume and os.path.exists(state_path):
        try:
            state = load_job(state_path)
            log_fn(f"↩️ Resuming job {state.job_id}")
        except (OSError, ValueError) as exc:
            # An unreadable state file must not block a fresh run; say so
            # loudly, because the user is about to redo work they may think
            # is already done.
            log_fn(f"⚠️ Ignoring unusable job state ({exc}); starting fresh")
            state = None
    if state is None:
        state = JobState(
            job_id=job_id or time.strftime("%Y%m%d-%H%M%S"),
            root=dest_root,
            created=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )

    state.music = music_path or state.music
    state.script = script_path or state.script
    for name in STAGE_ORDER:
        state.stage(name)
    save_job(state, state_path)

    # Stages that will genuinely run, for honest progress fractions: a bar that
    # counts skipped stages jumps to 80% instantly and tells the user nothing.
    planned = [STAGE_INGEST if card is not None else None,
               STAGE_MUSIC if music_path else None,
               STAGE_HIGHLIGHT,
               STAGE_COMBINE]
    planned = [p for p in planned if p]
    done_count = 0

    def announce(stage: Stage, status: str, detail: str = "") -> None:
        stage.status = status
        if detail:
            stage.detail = detail
        if stage_fn is not None:
            stage_fn(stage.name, status, detail)
        save_job(state, state_path)

    def advance(detail: str = "") -> None:
        if progress_fn is not None and planned:
            progress_fn(min(1.0, done_count / len(planned)), detail)

    def run_stage(name: str, fn, *, optional: bool = False, key: str = ""):
        """Execute one stage with uniform skip/timing/error handling.

        Returns the stage's value, or None when skipped or failed. A failing
        optional stage records the error and lets the pipeline continue; a
        failing required stage stops it. ``key`` fingerprints the settings the
        stage's output depends on, so changing them re-runs it.
        """
        nonlocal done_count
        stage = state.stage(name)
        _check_cancel(cancel_check)

        # Compared against the key recorded by the *previous* run, then
        # overwritten below — reversing those two lines makes every key match
        # itself and the check does nothing.
        reusable = resume and stage.satisfied_by(key)
        stage.key = key

        if reusable:
            log_fn(f"↩️ {name}: already done, skipping")
            done_count += 1
            advance(f"{name} (cached)")
            if stage_fn is not None:
                stage_fn(name, SKIPPED, stage.detail)
            return None

        announce(stage, RUNNING)
        advance(name)
        t0 = time.time()
        try:
            value = fn(stage)
        except PipelineCancelled:
            announce(stage, PENDING, "cancelled")
            raise
        except Exception as exc:
            stage.seconds = time.time() - t0
            stage.error = f"{type(exc).__name__}: {exc}"
            state.errors.append(f"{name}: {stage.error}")
            announce(stage, FAILED)
            log_fn(f"❌ {name} failed: {stage.error}")
            print(traceback.format_exc())   # debug log only; see CLAUDE.md
            if optional:
                return None
            raise
        stage.seconds = time.time() - t0
        announce(stage, DONE)
        done_count += 1
        advance(f"{name} done")
        return value

    # --- ingest ------------------------------------------------------------
    def _ingest(stage: Stage):
        from modules.gopro_ingest import ingest as ingest_card, write_manifest

        def ingest_progress(done: int, total: int, name: str) -> None:
            if progress_fn is not None and planned and total:
                base = done_count / len(planned)
                span = 1.0 / len(planned)
                progress_fn(min(1.0, base + span * (done / total)),
                            f"Copying {name}")

        result = ingest_card(card, dest_root, folder_name=folder_name,
                             verify=verify, log_fn=log_fn,
                             progress_fn=ingest_progress,
                             cancel_check=cancel_check)
        manifest = write_manifest(result)
        state.clips = result.paths
        stage.outputs = [manifest]
        stage.detail = f"{len(result.paths)} clip(s)"
        return result

    if card is not None:
        try:
            run_stage(STAGE_INGEST, _ingest)
        except PipelineCancelled:
            raise
        if not state.clips:
            # Resumed run: the manifest is the record of what landed.
            manifest = state.stage(STAGE_INGEST).outputs
            if manifest and os.path.exists(manifest[0]):
                from modules.gopro_ingest import read_manifest
                state.clips = [f["dest"] for f in read_manifest(manifest[0])["files"]]
    else:
        state.clips = list(source_paths or [])
        state.stage(STAGE_INGEST).status = SKIPPED

    if not state.clips:
        raise RuntimeError("no source clips to work with")

    # --- music analysis ----------------------------------------------------
    def _music(stage: Stage):
        from modules.music_analysis import analyze_music, save_analysis

        analysis = analyze_music(music_path, log_fn=log_fn)
        out = os.path.join(dest_root, "music_analysis.json")
        save_analysis(analysis, out)
        state.music_analysis = out
        stage.outputs = [out]
        stage.detail = f"{analysis.bpm:.1f} BPM, {len(analysis.beats)} beats"
        log_fn(f"🎵 {os.path.basename(music_path)}: {stage.detail}")
        return analysis

    analysis = None
    if music_path:
        # Optional: a music file we cannot analyse must not cost the user the
        # film. The reel still gets the track muxed on at the end.
        analysis = run_stage(STAGE_MUSIC, _music, optional=True,
                             key=os.path.basename(music_path))
        if analysis is None:
            # A skipped stage returns nothing, so on a resumed run the beat
            # grid has to be read back off disk. Without this, quantising is
            # silently skipped on exactly the runs most likely to want it —
            # the second and later attempts at the same film.
            saved = state.stage(STAGE_MUSIC).outputs
            if saved and os.path.exists(saved[0]):
                try:
                    from modules.music_analysis import load_analysis
                    analysis = load_analysis(saved[0])
                    state.music_analysis = saved[0]
                    log_fn(f"↩️ Reusing the beat grid: {analysis.bpm:.1f} BPM")
                except Exception as exc:
                    log_fn(f"⚠️ Saved beat grid unusable ({exc}); "
                           f"cuts will not be quantised")

    # --- script ------------------------------------------------------------
    script = None
    if script_path:
        try:
            from modules.script_plan import load_script
            script = load_script(script_path)
            log_fn(f"📝 Script '{script.title}': {script.clip_count} clip(s), "
                   f"target {script.target_duration:.0f}s")
        except Exception as exc:
            # A malformed script is a user error worth surfacing, but it must
            # not strand footage that is already copied and analysable.
            log_fn(f"⚠️ Script ignored ({type(exc).__name__}: {exc})")
            state.errors.append(f"script: {exc}")

    # --- highlights --------------------------------------------------------
    def _highlight(stage: Stage):
        gui_config = dict(config or {})
        if script is not None:
            gui_config = apply_script_to_config(script, gui_config)

        runner = highlight_runner or _default_highlight_runner

        def stage_progress(*args) -> None:
            # The engine's progress signature varies by call site; only the
            # detail text is reliably useful here.
            detail = next((a for a in args if isinstance(a, str)), "")
            if progress_fn is not None and planned:
                progress_fn(min(1.0, done_count / len(planned)), detail or "Analysing")

        class _CancelFlag:
            """Adapts ``cancel_check()`` to the ``.is_set()`` flag the engine
            expects — it takes a threading/multiprocessing Event, and this
            pipeline takes a callable."""

            def is_set(self) -> bool:
                return bool(cancel_check and cancel_check())

        results = runner(state.clips, gui_config, log_fn, stage_progress, _CancelFlag())

        outputs: list[str] = []
        if isinstance(results, str):
            outputs = [results]
        else:
            for item in results or []:
                out = item[1] if isinstance(item, (list, tuple)) and len(item) > 1 else item
                if out:
                    outputs.append(out)
        outputs = [p for p in outputs if p and os.path.exists(p)]
        if not outputs:
            raise RuntimeError("the engine produced no highlight clips")
        state.highlights = outputs
        stage.outputs = list(outputs)
        stage.detail = f"{len(outputs)} highlight(s)"
        return outputs

    run_stage(STAGE_HIGHLIGHT, _highlight)
    if not state.highlights:
        state.highlights = [p for p in state.stage(STAGE_HIGHLIGHT).outputs
                            if os.path.exists(p)]

    # --- combine -----------------------------------------------------------
    def _combine(stage: Stage):
        from modules.edl import (edl_from_clips, quantise_to_music, render_edl,
                                 save_edl)

        reel = os.path.join(dest_root, output_name)
        stem = os.path.splitext(output_name)[0]
        edl_out = os.path.join(dest_root, f"{stem}.edl.yaml")

        # The cut list is written before the render, not after: it is the
        # document that explains what is about to happen, and it has to survive
        # a render that fails halfway.
        cut_list = edl_from_clips(
            state.highlights,
            title=(script.title if script is not None else stem),
            transition=transition, transition_duration=transition_duration)
        cut_list.width, cut_list.height = int(width), int(height)
        cut_list.fps, cut_list.crf = int(fps), int(crf)
        if music_path:
            cut_list.music = music_path
            cut_list.music_mode = music_mode
            cut_list.music_volume = float(music_volume)

        if quantise and analysis is not None:
            cut_list = quantise_to_music(cut_list, analysis, unit=quantise,
                                         transition_bars=transition_bars,
                                         log_fn=log_fn)
        elif quantise:
            log_fn("⚠️ Beat quantising asked for, but no music analysis — skipped")

        save_edl(cut_list, edl_out)
        state.edl = edl_out
        log_fn(f"📝 Cut list: {edl_out} "
               f"({len(cut_list.cuts)} cuts, {cut_list.duration:.0f}s)")

        untouched = (
            len(cut_list.cuts) == 1
            and not music_path
            and not quantise
            and abs(cut_list.cuts[0].start) < 0.01
        )
        if untouched:
            # One clip, kept whole, with nothing to lay over it: the clip is
            # already the film. Re-cutting and re-encoding it would cost a
            # generation of quality to produce the same frames.
            import shutil
            if os.path.abspath(cut_list.cuts[0].source) != os.path.abspath(reel):
                shutil.copy2(cut_list.cuts[0].source, reel)
            log_fn("🎬 One clip, kept whole — copied rather than re-encoded")
        else:
            # render_edl re-cuts each clip at its (possibly quantised)
            # timestamps and joins with transitions. Music is applied there
            # rather than in a later stage, because that stage is already
            # re-encoding and a second pass over the whole reel would be waste.
            # music_optional: the expensive work is everything before this, and
            # an audio filter must not be able to destroy it.
            render_edl(cut_list, reel, music_optional=True, log_fn=log_fn,
                       cancel_check=cancel_check)

        state.reel = reel
        state.final = reel
        stage.outputs = [reel, edl_out]
        stage.detail = f"{len(cut_list.cuts)} cuts, {cut_list.duration:.0f}s"
        return reel

    # Everything the finished reel's shape depends on. Changing any of it must
    # rebuild, even though film.mp4 is sitting right there.
    render_key = "|".join(str(v) for v in (
        transition, transition_duration, transition_bars, quantise,
        width, height, fps, crf, music_path, music_mode, music_volume,
        output_name, len(state.highlights)))

    run_stage(STAGE_COMBINE, _combine, key=render_key)
    if not state.reel:
        outs = state.stage(STAGE_COMBINE).outputs
        state.reel = outs[0] if outs else ""
    state.final = state.reel

    # The music is laid inside the combine stage now: that stage already
    # re-encodes everything to build the transitions, so muxing there costs
    # nothing, while a separate pass would rewrite the whole reel a second
    # time. The stage name survives for job files written by older builds.
    state.stage(STAGE_MUSIC_MIX).status = SKIPPED

    save_job(state, state_path)
    seconds = time.time() - started
    ok = not any(s.status == FAILED for s in state.stages.values())
    log_fn(f"{'✅' if ok else '⚠️'} Job {state.job_id} finished in {seconds:.0f}s "
           f"-> {state.final or '(no output)'}")
    if progress_fn is not None:
        progress_fn(1.0, "done")
    return JobResult(state=state, output=state.final, seconds=seconds, ok=ok)


def apply_script_to_config(script, config: dict) -> dict:
    """Fold a :class:`modules.script_plan.Script` into an engine config dict.

    The engine is driven by a flat config (the same one the Qt GUI and the
    sidecar build), so a script has to be expressed in that vocabulary rather
    than as a parallel control path. What survives the translation is what the
    engine can actually act on: the union of every beat's match terms becomes
    the interest lists, and the script's total becomes the duration budget.

    What deliberately does *not* survive is per-beat ordering and per-beat
    duration. The current selector ranks the whole video globally and has no
    notion of "this clip belongs to beat 3", so pretending otherwise here would
    produce a config that looks like it honours the script and does not. Those
    directives are carried by ``compile_directives`` for the caller that can
    use them; see docs/AUTO-PIPELINE.md.

    The input config is not mutated — callers reuse it across runs.
    """
    merged = dict(config or {})

    def extend(section: str, key: str, values) -> None:
        if not values:
            return
        block = dict(merged.get(section) or {})
        existing = list(block.get(key) or [])
        for value in values:
            if value not in existing:
                existing.append(value)
        block[key] = existing
        merged[section] = block

    objects, actions, keywords = [], [], []
    for beat in script.beats:
        objects += list(beat.objects)
        actions += list(beat.actions)
        keywords += list(beat.keywords)

    extend("objects", "interesting", objects)
    extend("actions", "interesting", actions)
    extend("keywords", "interesting", keywords)

    if script.target_duration:
        highlights = dict(merged.get("highlights") or {})
        highlights["max_duration"] = script.target_duration
        merged["highlights"] = highlights

    return merged
