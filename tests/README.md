# VideoHighlighter — Test Suite

Phase 0 safety net per `docs/IMPLEMENTATION-ROADMAP.md`.

## What this suite is for

Locking down the **pure, deterministic logic** of the pipeline so the upcoming
`pipeline.py` / `main.py` refactor cannot silently break highlight selection,
forbidden-range avoidance, or subtitle output. These are the modules where a
regression would not show up as a crash — it would show up as "highlights
slightly different from before", which is much harder to notice.

## What this suite is NOT

It does **not** exercise:

- the ML models (YOLO, OpenVINO action recognition, Whisper, Resemblyzer)
- GPU / hardware-dependent paths
- the GUI (`main.py`)
- FFmpeg cutting / concat

Those need an integration-test layer with sample videos, which is a separate
investment (see roadmap Tier 1 #1 detail: a 20-video golden-output corpus).
This suite is the precondition for that work.

## How to run

```powershell
# From videohighlighter/ root:
python -m venv .venv-test
.\.venv-test\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pip install numpy        # the only real heavy dep the tests need
pytest
```

Expected: all green in <5 seconds.

## How it stays light

`tests/conftest.py` installs `MagicMock` shims for `cv2`, `torch`, `whisper`,
`googletrans`, `ultralytics`, `openvino`, and friends **before** any test
imports the modules under test. So `pip install -r requirements.txt`
(~1.9 GB) is not required to run the suite — only pytest + numpy.

If you have a full dev environment installed, the real libraries take
precedence over the shims and nothing changes.

## Adding tests

Three rules for tests added under this directory:

1. **No new heavy deps without shimming them in `conftest.py` first.** The
   "5 MB install, 5 second runtime" property is the whole point.
2. **Pin behaviour, do not specify desired behaviour.** Characterization tests
   document what the code does *today* so refactor cannot drift. If you find
   a bug while writing a test, write the test that pins the bug, fix the bug
   in a separate commit, then update the test.
3. **Name the function under test in the file name.** `test_<function>.py`.
   Easy `grep`, easy file-to-test mapping.

## Coverage today

| File | Module covered | Why it matters |
|------|---------------|----------------|
| `test_merge_seconds.py` | `modules.compute_forbidden._merge_seconds` | Powers the AVOID(skip) flow |
| `test_cluster_points.py` | `modules.auto_segments.cluster_points` | Auto-segmentation core when `CLIP_TIME=0` |
| `test_snap_to_scene.py` | `modules.auto_segments.snap_to_scene` | Scene-boundary alignment |
| `test_region.py` | `modules.auto_segments.Region` | Overlap/merge geometry for highlight regions |
| `test_srt_timestamp.py` | `modules.transcript_srt.format_timestamp_srt` | SRT spec compliance for every subtitle written |

Next additions (Phase 1+):

- `modules.translate` — adapter contract: signature stable across NLLB / MarianMT / googletrans backends.
- `modules.logging_utils` — JSON-schema validation of emitted log records.
- Scoring math (the per-second weighted sum + multi-signal boost) — currently buried in `pipeline.run_pipeline`; needs extraction during Phase 2 refactor first.
