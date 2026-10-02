"""Where does action recognition's time go, outside the model?

    python tools/teach_lab/bench_app.py <video> --src <code checkout> --variant pipeline|no_annot|no_person|decode

Runs the code in --src (e.g. a 0.12.1 checkout, to sit next to the 0.12.1 exe)
headless, with the call pipeline.py makes for "Intel only" (sample_rate 5,
person detection on):

  no_annot   that call as the app makes it by default
  pipeline   the same with the annotated video written -- what the app does
             only when "draw action labels" (draw_action_labels) is on
  no_person  no_annot without person detection (whole frame to the encoder)
  decode     no models at all: read the video the way it is read for analysis,
             every frame decoded -- the floor any analysis pays

Prints the function's own PERFORMANCE SUMMARY plus the wall time.
"""
import argparse
import os
import sys
import tempfile
import time


def decode_only(video, sample_rate):
    import cv2
    cap = cv2.VideoCapture(video)
    n = kept = 0
    t0 = time.perf_counter()
    while True:
        ok = cap.grab()
        if not ok:
            break
        if n % sample_rate == 0:
            ok, _ = cap.retrieve()
            kept += 1
        n += 1
    dt = time.perf_counter() - t0
    print(f"decode only: {n} frames ({kept} retrieved) in {dt:.1f}s -> {n / dt:.0f} video fps")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--src", required=True)
    ap.add_argument("--variant", default="pipeline", choices=["pipeline", "no_annot", "no_person", "decode"])
    ap.add_argument("--sample-rate", type=int, default=5)
    args = ap.parse_args()

    if args.variant == "decode":
        return decode_only(args.video, args.sample_rate)

    sys.modules.setdefault("sentence_transformers", None)
    sys.path.insert(0, args.src)
    os.chdir(args.src)
    import action_recognition as ar
    # the device pipeline.py hands it: the compute preference, not OpenVINO's AUTO
    from modules.system.device_utils import detect_best_device
    device = getattr(detect_best_device(log_fn=lambda *_: None), "openvino_device", "AUTO") or "AUTO"
    print(f"code: {args.src} ({ar.__file__}), OpenVINO device {device}", flush=True)

    annotated = None
    if args.variant != "no_annot":
        annotated = os.path.join(tempfile.gettempdir(), f"bench_annot_{os.getpid()}.mp4")
    t0 = time.perf_counter()
    ar.run_action_detection(
        video_path=args.video, sample_rate=args.sample_rate, debug=False,
        interesting_actions=None, draw_bboxes=args.variant != "no_annot",
        annotated_output=annotated, use_person_detection=args.variant != "no_person",
        max_people=2, include_model_type=False, enable_r3d=False,
        action_models="intel_only", device=device,
    )
    print(f"\nwall time: {time.perf_counter() - t0:.1f}s ({args.variant})", flush=True)
    if annotated and os.path.exists(annotated):
        os.remove(annotated)


if __name__ == "__main__":
    sys.exit(main())
