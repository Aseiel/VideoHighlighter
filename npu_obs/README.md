# npu_obs — object detection on the Intel NPU, next to OBS

A small standalone tool that runs VideoHighlighter's YOLOX detector on the
**NPU** of an Intel Core Ultra processor ("Intel AI Boost"). The game, OBS and
its encoder keep the CPU and GPU.

It can:

- **`probe`**: check that OpenVINO sees the NPU and time the detector on it.
- **`video`**: scan a finished recording and list the time spans where each
  class is on screen.
- **`live`**: watch a running OBS (or a capture device) and log those spans as
  they happen. While OBS records, each span is stamped with its position in the
  recording, and it can optionally add a recording chapter.

The detector lives in `modules/vision/npu_detector.py`. It is the app's own
`YoloxOpenVINODetector` with the pre-processing, decoding and NMS inherited
unchanged, compiled for the NPU. Any YOLOX-layout model works with it: the stock
COCO models, or one you taught the app yourself (`--model`).

## Measured, not promised

These are the numbers on a Core Ultra 7 265K (NPU "Intel AI Boost"),
OpenVINO 2026.2, from `python -m npu_obs probe --compare`:

| model      | device | inference | CPU time per inference |
|------------|--------|-----------|------------------------|
| yolox_nano | NPU    | 4.5 ms    | < 1 ms                 |
| yolox_nano | CPU    | 3.9 ms    | 24 ms                  |
| yolox_tiny | NPU    | 5.3 ms    | < 1 ms                 |
| yolox_tiny | CPU    | 9.8 ms    | 68 ms                  |
| yolox_s    | NPU    | 12.6 ms   | 1 ms                   |
| yolox_s    | CPU    | 33 ms     | 235 ms                 |

- **Accuracy:** NPU detections match the CPU's. On 5 gameplay frames and 48
  game screenshots at a 0.1 threshold, 846 of 851 CPU boxes appear on the NPU with
  IoU > 0.9 and confidence within 0.016. The few that differ sit right at the
  threshold, because the NPU computes in FP16.
- **Whole frame:** resize, letterbox, decode and NMS still run on the CPU, about
  2–4 ms for a 1080p–1440p frame. The uint8→float conversion and the HWC→CHW
  transpose were moved into the compiled model, so the NPU does them.
- **Recording scan:** a 48-minute 1440p60 H.264 OBS recording, sampled every
  0.5 s with yolox_s (5,801 samples), took **95 s, about 30× real time**.
  Decoding used the GPU's video engine, and the process used about 1.6 CPU
  cores. `--cpu-decode` took the same time (99 s) on about 3.5 cores.
- **Start-up:** the compiled model is cached under `cache/openvino/`, so after
  the first run the NPU loads in under 0.1 s.

## Setup

```bash
pip install openvino opencv-python numpy psutil
pip install websocket-client          # only for `live` with OBS
python tools/get_yolox_model.py nano tiny s
python -m npu_obs probe
```

The NPU needs the Intel NPU driver. If `probe` lists no `NPU` device, install
or update it from Intel. Everything still runs without it, on the CPU, and the
tool says so.

## Scanning a recording

```bash
python -m npu_obs video "E:\recordings\session.mp4"
python -m npu_obs video session.mp4 --classes person,dog --every 1 --out session.json
python -m npu_obs video session.mp4 --model models/custom/my_model/my_model.xml
```

The scan writes `<name>.npu.json` with the spans and a per-class summary:

```json
{
  "spans":   [{"label": "person", "start": 12.5, "end": 31.0, "peak": 0.91, "hits": 38}],
  "summary": {"person": {"spans": 4, "seconds": 96.5, "peak": 0.93}}
}
```

`--frames samples.jsonl` also writes every sample's boxes. A span opens after
`--min-hits` samples in a row (default 2) and closes once the class has been
gone for longer than `--max-gap`, so a box that drops out for a single sample
does not split a span.

## Watching OBS live

1. In OBS, open **Tools → WebSocket Server Settings**, tick **Enable WebSocket
   server**, and copy the password (**Show Connect Info**).
2. Run:

```bash
set OBS_WEBSOCKET_PASSWORD=...        # or --obs-password
python -m npu_obs live                 # watches the program scene, 4x a second
python -m npu_obs live --classes person --chapters --preview
```

Each sample asks OBS for a screenshot of the program scene (`--obs-source`
picks another scene or source). OBS scales it down to the detector's input
width on the GPU, so only a small JPEG reaches this process. Events go to
`npu_live_<time>.jsonl`:

```json
{"event": "start", "label": "person", "start": 81.2, "end": 81.7, "peak": 0.88, "record_ms": 754300, ...}
```

`record_ms` is the span's position in the file OBS is recording, so it lines up
with that file afterwards. `--chapters` also adds an OBS recording chapter
named after the class when a span starts. That needs OBS 30.2+ recording to
Hybrid MP4; otherwise you get one warning and the events are still logged.

Without obs-websocket, `--capture 1` reads a capture device instead, such as
the **OBS Virtual Camera** (Start Virtual Camera in OBS, then find its index).

**What this costs the game:** the detector runs on the NPU. Each screenshot
costs OBS one GPU readback and a JPEG encode, 4 times a second by default
(`--interval`). The tool runs at below-normal process priority
(`--normal-priority` to turn that off).

## Notes and limits

- **The stock models know COCO's 80 everyday classes** (people, vehicles,
  animals, household objects). Rendered or stylised footage often contains none
  of them. To find what matters in your footage, teach the app a model of your
  own and pass it with `--model`. Any YOLOX-layout `.xml` or `.onnx` works, with
  class names read from the model or from a `labels.json` beside it.
- **GPU and CPU decoding can differ by about one colour level.** When a file has
  no colour-space tag, the two decoders pick different conversion matrices. That
  can flip a detection sitting right at the threshold, and nothing else.
- Sample times are each frame's own timestamp. OBS records at a variable frame
  rate (the recording above averaged 29 fps, with frame gaps from 17 to 50 ms),
  so counting frames would drift.
