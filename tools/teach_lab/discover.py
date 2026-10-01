"""Group person-focused crops by what they look like, how the bodies are laid
out, and how they move -- proposals for classes, before anyone names them.

    python tools/teach_lab/discover.py <crops folder> <out folder> [--timeline timeline.csv]
        [--samples <uncropped samples folder> ...] [--weights clip=1,pose=.8,motion=.4,scene=.8]

Three descriptions per crop, each cached in <out>/features.npz so a re-run only
reads new crops:

  clip    what is in the picture (CLIP, mean of 4 frames)
  pose    YOLOX people + RTMPose keypoints on 6 frames: how many people, how
          each torso is tilted, how they sit relative to each other, the main
          person's skeleton normalised to its own torso
  motion  how much the picture changes and at what rhythm (48 frames, grey 64px)

The three are standardised, weighted and clustered (KMeans on a PCA of the
mix -- the crops form a continuum with no density gaps, so HDBSCAN found one
blob; every crop gets its nearest group instead). Groups are numbered tightest
first, and each group's crops are ordered most typical first. Each group gets a folder of hard
links to its crops, a contact sheet, and -- when a timeline.csv from
`teach from-dataset` is given -- the class guesses its samples already had.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

# COCO-17 layout (what RTMPose body7 returns).
NOSE, L_SH, R_SH, L_HIP, R_HIP = 0, 5, 6, 11, 12
KP_CONF = 0.3
MIN_KEYPOINTS = 3          # a proposal with fewer visible keypoints is not a person
PERSON_DIMS = 4 + 34 + 1
POSE_DIMS = 2 * (1 + 2 * PERSON_DIMS + 4)
POSE_FRAMES = 6
CLIP_FRAMES = 4
MOTION_FRAMES = 48
WEIGHTS = {"clip": 1.0, "pose": 0.8, "motion": 0.4, "scene": 0.8}
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]


# ---------------------------------------------------------------- reading

def frames_at(path: str, count: int, consecutive: bool = False, size=None) -> list:
    cap = cv2.VideoCapture(path)
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            return []
        out = []
        if consecutive:
            start = max(0, (total - count) // 2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            for _ in range(min(count, total)):
                ok, f = cap.read()
                if not ok:
                    break
                out.append(cv2.resize(f, size) if size else f)
            return out
        for k in range(count):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int((k + 0.5) * total / count))
            ok, f = cap.read()
            if ok:
                out.append(cv2.resize(f, size) if size else f)
        return out
    finally:
        cap.release()


# ---------------------------------------------------------------- features

def _mid(kp, a, b):
    pa, pb = kp[a], kp[b]
    if pa[2] > KP_CONF and pb[2] > KP_CONF:
        return (pa[:2] + pb[:2]) / 2
    if pa[2] > KP_CONF:
        return pa[:2]
    if pb[2] > KP_CONF:
        return pb[:2]
    return None


def person_vector(kp, box, w, h):
    """Torso tilt, size, visibility and the skeleton in torso units (39 dims).

    A close crop often cuts shoulders or hips off; then the skeleton is laid
    out relative to the person's box instead, and the tilt is left at zero --
    ``torso_ok`` (last value) says which of the two it is.
    """
    sh, hip = _mid(kp, L_SH, R_SH), _mid(kp, L_HIP, R_HIP)
    seen = kp[:, 2] > KP_CONF
    vis = float(seen.mean())
    if sh is not None and hip is not None:
        torso = sh - hip
        length = float(np.linalg.norm(torso)) or 1.0
        angle = np.arctan2(torso[1], torso[0])
        head = [np.sin(angle), np.cos(angle), length / max(w, h), vis]
        origin, unit_len, ok = hip, length, 1.0
    else:
        x1, y1, x2, y2 = box
        origin = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        unit_len = max(y2 - y1, x2 - x1, 1.0) / 2
        head = [0.0, 0.0, unit_len * 2 / max(w, h), vis]
        ok = 0.0
    rel = (kp[:, :2] - origin) / unit_len
    rel[~seen] = 0.0
    return np.concatenate([head, np.clip(rel, -4, 4).ravel(), [ok]]).astype(np.float32)


def pose_features(frames, detector, estimator):
    """Mean and spread over frames of: people count, the two largest people's
    torso vectors, and how the two sit relative to each other."""
    rows = []
    for f in frames:
        h, w = f.shape[:2]
        # Propose generously, keep what RTMPose finds a body in (as the cropper does).
        cands = [d for d in detector.detect(f) if d.class_id == 0]
        cands.sort(key=lambda d: (d.x2 - d.x1) * (d.y2 - d.y1), reverse=True)
        cands = cands[:5]
        found = estimator.estimate(f, [(d.x1, d.y1, d.x2, d.y2) for d in cands]) if cands else []
        kept = [(d, p) for d, p in zip(cands, found)
                if int((p.keypoints[:, 2] > KP_CONF).sum()) >= MIN_KEYPOINTS][:2]
        people = [d for d, _ in kept]
        poses = [p for _, p in kept]
        vecs = [person_vector(p.keypoints, (d.x1, d.y1, d.x2, d.y2), w, h) for d, p in kept]
        a = vecs[0] if vecs else np.zeros(PERSON_DIMS, np.float32)
        b = vecs[1] if len(vecs) > 1 else np.zeros(PERSON_DIMS, np.float32)
        rel = np.zeros(4, np.float32)
        if len(poses) == 2:
            ka, kb = poses[0].keypoints, poses[1].keypoints
            scale = max(w, h)
            for i, (x, y) in enumerate(((ka, kb), (kb, ka))):
                head, hip = x[NOSE], _mid(y, L_HIP, R_HIP)
                if head[2] > KP_CONF and hip is not None:
                    rel[i] = np.linalg.norm(head[:2] - hip) / scale
            ba, bb = people[0], people[1]
            ix = max(0, min(ba.x2, bb.x2) - max(ba.x1, bb.x1))
            iy = max(0, min(ba.y2, bb.y2) - max(ba.y1, bb.y1))
            inter = ix * iy
            union = (ba.x2 - ba.x1) * (ba.y2 - ba.y1) + (bb.x2 - bb.x1) * (bb.y2 - bb.y1) - inter
            rel[2] = inter / union if union else 0
            rel[3] = 1.0
        rows.append(np.concatenate([[len(people) / 2.0], a, b, rel]))
    if not rows:
        return np.zeros(POSE_DIMS, np.float32)
    rows = np.stack(rows)
    return np.concatenate([rows.mean(0), rows.std(0)]).astype(np.float32)


def motion_features(frames):
    if len(frames) < 3:
        return np.zeros(6, np.float32)
    g = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames])
    d = np.abs(np.diff(g, axis=0)).mean(axis=(1, 2))
    spec = np.abs(np.fft.rfft(d - d.mean()))
    freqs = np.fft.rfftfreq(len(d), d=1 / 25.0)
    peak = int(np.argmax(spec[1:]) + 1) if len(spec) > 1 else 0
    periodicity = float(spec[peak] / (spec[1:].sum() + 1e-6)) if peak else 0.0
    # where in the picture it moves: centre vs edges
    m = np.abs(np.diff(g, axis=0)).mean(0)
    centre = m[16:48, 16:48].mean() / (m.mean() + 1e-6)
    return np.array([d.mean(), d.std(), freqs[peak] if peak else 0.0, periodicity,
                     centre, np.percentile(d, 90)], np.float32)


# ---------------------------------------------------------------- main

def sample_of(crop_name: str) -> str:
    return re.match(r"(v\d+__\d+)", crop_name).group(1)


def read_scenes(args, crops) -> dict:
    """Pose of the whole, uncropped sample each crop came from.

    A crop is often too close for a body to be found in it; the sample it was
    cut from still shows who is where. Cached in <features>/scene.npz.
    """
    if not args.samples:
        return {}
    path = os.path.join(args.features or args.out, "scene.npz")
    scene = {}
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        scene = dict(zip(z["names"], z["pose"]))
    wanted = {sample_of(os.path.basename(c)) for c in crops}
    sources = {}
    for folder in args.samples:
        for p in glob.glob(os.path.join(folder, "*.mp4")):
            sid = os.path.splitext(os.path.basename(p))[0]
            if sid in wanted and sid not in scene:
                sources[sid] = p
    if not sources:
        return scene
    from modules.vision.detection_backend import YoloxOpenVINODetector, find_default_yolox_ir
    from modules.vision.pose_backend import build_pose_estimator

    detector = YoloxOpenVINODetector(find_default_yolox_ir(prefer="large"),
                                     class_names=["person"], device=args.device, score_thr=0.05)
    estimator = build_pose_estimator(device=args.device)
    t0 = time.time()
    for i, (sid, p) in enumerate(sorted(sources.items()), 1):
        try:
            scene[sid] = pose_features(frames_at(p, POSE_FRAMES), detector, estimator)
        except Exception as exc:  # noqa: BLE001
            print(f"discover: {sid}: {type(exc).__name__}: {exc}", flush=True)
        if i % 100 == 0 or i == len(sources):
            print(f"discover: scenes {i}/{len(sources)} ({(time.time() - t0) / i:.2f}s each)", flush=True)
            np.savez(path, names=np.array(list(scene)), pose=np.stack(list(scene.values())))
    return scene


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("crops")
    ap.add_argument("out")
    ap.add_argument("--timeline", help="timeline.csv from teach from-dataset")
    ap.add_argument("--groups", type=int, default=20)
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--features-only", action="store_true")
    ap.add_argument("--cached-only", action="store_true", help="group what has features; read nothing new")
    ap.add_argument("--weights", default="", help="e.g. clip=0,pose=1,motion=0.5")
    ap.add_argument("--features", help="read features.npz from here (default: <out>)")
    ap.add_argument("--samples", nargs="*", default=[], help="folders of the uncropped samples, for scene layout")
    args = ap.parse_args(argv)
    # These tools never use sentence-transformers; optimum.intel imports it
    # whenever it is installed, and a broken install (e.g. a torchcodec that
    # cannot load) would take CLIP down with it.
    sys.modules.setdefault("sentence_transformers", None)

    os.makedirs(args.out, exist_ok=True)
    weights = dict(WEIGHTS)
    for part in filter(None, args.weights.split(",")):
        k, v = part.split("=")
        weights[k.strip()] = float(v)
    cache_path = os.path.join(args.features or args.out, "features.npz")
    cache = {}
    if os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=True)
        cache = {n: (c, p, m) for n, c, p, m in zip(z["names"], z["clip"], z["pose"], z["motion"])}

    crops = sorted(glob.glob(os.path.join(args.crops, "*.mp4")))
    todo = [] if args.cached_only else [c for c in crops if os.path.basename(c) not in cache]
    print(f"discover: {len(crops)} crops, {len(todo)} to read", flush=True)
    if todo:
        from modules.teach.embed import ClipBackend, unit
        from modules.vision.detection_backend import YoloxOpenVINODetector, find_default_yolox_ir
        from modules.vision.pose_backend import build_pose_estimator

        clip = ClipBackend()
        detector = YoloxOpenVINODetector(find_default_yolox_ir(prefer="large"),
                                         class_names=["person"], device=args.device,
                                         score_thr=0.05)
        estimator = build_pose_estimator(device=args.device)
        if estimator is None:
            raise SystemExit("no RTMPose model: python tools/get_rtmpose_model.py")
        t0 = time.time()
        for i, path in enumerate(todo, 1):
            name = os.path.basename(path)
            try:
                cf = frames_at(path, CLIP_FRAMES)
                c = unit(unit(clip.images(cf)).mean(0)) if cf else np.zeros(512, np.float32)
                p = pose_features(frames_at(path, POSE_FRAMES), detector, estimator)
                m = motion_features(frames_at(path, MOTION_FRAMES, consecutive=True, size=(64, 64)))
                cache[name] = (c, p, m)
            except Exception as exc:  # noqa: BLE001 - one bad crop must not stop the rest
                print(f"discover: {name}: {type(exc).__name__}: {exc}", flush=True)
            if i % 50 == 0 or i == len(todo):
                print(f"discover: read {i}/{len(todo)} ({(time.time() - t0) / i:.2f}s each)", flush=True)
                names = list(cache)
                np.savez(cache_path, names=np.array(names),
                         clip=np.stack([cache[n][0] for n in names]),
                         pose=np.stack([cache[n][1] for n in names]),
                         motion=np.stack([cache[n][2] for n in names]))
    scene = read_scenes(args, crops)
    if args.features_only:
        return 0

    names = [os.path.basename(c) for c in crops if os.path.basename(c) in cache]
    blocks = {"clip": np.stack([cache[n][0] for n in names]),
              "pose": np.stack([cache[n][1] for n in names]),
              "motion": np.stack([cache[n][2] for n in names])}
    if scene:
        blank = np.zeros(POSE_DIMS, np.float32)
        blocks["scene"] = np.stack([scene.get(sample_of(n), blank) for n in names])
        print(f"discover: scene layout for {sum(sample_of(n) in scene for n in names)}"
              f"/{len(names)} crops", flush=True)
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    mixed = []
    for key, x in blocks.items():
        x = StandardScaler().fit_transform(x)
        x = x / np.sqrt(x.shape[1])          # every block counts by its weight, not its width
        if weights[key] > 0:
            mixed.append(x * np.sqrt(weights[key]))
    x = np.concatenate(mixed, axis=1)
    x = PCA(n_components=min(32, len(names) - 1), random_state=0).fit_transform(x)
    km = KMeans(n_clusters=min(args.groups, len(names)), n_init=10, random_state=0).fit(x)
    labels = km.labels_
    # distance to the group's centre: how typical a crop is of its group
    dist = np.linalg.norm(x - km.cluster_centers_[labels], axis=1)

    guesses = {}
    if args.timeline and os.path.exists(args.timeline):
        with open(args.timeline, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                guesses[row["sample"]] = row

    # fresh group folders every run
    for old in glob.glob(os.path.join(args.out, "g*")):
        if os.path.isdir(old):
            for f in os.listdir(old):
                os.remove(os.path.join(old, f))
            os.rmdir(old)

    groups = defaultdict(list)
    for i, (n, lab) in enumerate(zip(names, labels)):
        groups[int(lab)].append((dist[i], n))
    # tightest groups first; inside a group, the most typical crop first
    spread = {g: float(np.median([d for d, _ in m])) for g, m in groups.items()}
    order = sorted(groups, key=lambda g: spread[g])
    rename = {g: f"g{i + 1:02d}" for i, g in enumerate(order)}
    groups = {g: [n for _, n in sorted(m)] for g, m in groups.items()}

    rows, summary = [], {}
    for g, members in groups.items():
        gname = rename[g]
        folder = os.path.join(args.out, gname)
        os.makedirs(folder, exist_ok=True)
        times, guess_counter = [], Counter()
        for n in members:
            src = os.path.join(args.crops, n)
            try:
                os.link(src, os.path.join(folder, n))
            except OSError:
                pass
            sample = sample_of(n)
            t = int(sample.split("__")[1]) / 1000.0  # noqa: E501
            times.append(t)
            ms = int(sample.split("__")[1])
            row = guesses.get(sample) or guesses.get(f"{sample.split('__')[0]}__{ms // 5000 * 5000:08d}", {})
            if row:
                guess_counter[row.get("best", "")] += 1
            rows.append({"crop": n, "sample": sample, "at_s": t, "group": gname,
                         "guess": row.get("best", ""), "guess_class": row.get("class", "")})
        contact_sheet(members[:36], args.crops, os.path.join(folder, "_sheet.jpg"))
        idx = [names.index(n) for n in members]
        pm = blocks["pose"][idx].mean(0)
        mm = blocks["motion"][idx].mean(0)
        summary[gname] = {
            "crops": len(members),
            "spread": round(spread[g], 2),
            "samples": len({r["sample"] for r in rows if r["group"] == gname}),
            "minutes": sorted({int(t // 60) for t in times}),
            "people_avg": round(float(pm[0] * 2), 2),
            "main_torso_upright": round(float(-pm[1]), 2),   # -sin: 1 = head up, 0 = lying
            "two_people_share": round(float(pm[POSE_DIMS // 2 - 1]), 2),
            "motion": round(float(mm[0]), 2),
            "rhythm_hz": round(float(mm[2]), 2),
            "earlier_guesses": dict(guess_counter.most_common(4)),
        }
    with open(os.path.join(args.out, "groups.csv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(sorted(rows, key=lambda r: (r["group"], r["crop"])))
    summary = dict(sorted(summary.items()))
    with open(os.path.join(args.out, "groups.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    overview(args.out, rows, summary)
    print(json.dumps({"crops": len(names), "groups": len(order), "per_group": summary}, indent=1))
    return 0


def overview(out_dir, rows, summary):
    """groups.html: the video as a strip coloured by group, then every group's sheet."""
    import colorsys
    import html

    names = [g for g in summary if g != "g_noise"]
    colour = {g: "#%02x%02x%02x" % tuple(int(c * 255) for c in colorsys.hsv_to_rgb(
        (i * 0.618) % 1, 0.55, 0.9)) for i, g in enumerate(names)}
    colour["g_noise"] = "#555"
    end = max((r["at_s"] for r in rows), default=0) + 5
    by_time = defaultdict(Counter)
    for r in rows:
        by_time[int(r["at_s"] // 2.5)][r["group"]] += 1
    cells = []
    for slot in range(int(end // 2.5) + 1):
        g = by_time[slot].most_common(1)[0][0] if by_time[slot] else ""
        t = slot * 2.5
        cells.append(f'<i style="background:{colour.get(g, "#111")}" '
                     f'title="{int(t // 60):02d}:{t % 60:04.1f} {g}"></i>')
    parts = [f"<section><h2 style='border-color:{colour[g]}'>{g} "
             f"<small>{s['crops']} crops · {s['samples']} samples · people {s['people_avg']} · "
             f"upright {s['main_torso_upright']} · two together {s['two_people_share']} · "
             f"motion {s['motion']} · rhythm {s['rhythm_hz']} Hz · minutes "
             f"{', '.join(map(str, s['minutes'][:20]))}</small></h2>"
             f"<p>earlier guesses: {html.escape(', '.join(f'{k} {v}' for k, v in s['earlier_guesses'].items()))}</p>"
             f"<img src='{g}/_sheet.jpg' loading='lazy'></section>"
             for g, s in summary.items()]
    page = ("<!doctype html><meta charset=utf-8><title>Groups</title><style>"
            "body{background:#111;color:#ddd;font:14px system-ui;margin:16px}"
            "#strip{display:flex;height:28px;margin:8px 0 24px}#strip i{flex:1}"
            "h2{border-left:8px solid;padding-left:8px;font-size:16px}small{color:#999;font-weight:normal}"
            "img{max-width:100%}p{color:#aaa;margin:4px 0 8px}</style>"
            f"<h1>{len(names)} groups</h1><div id=strip>{''.join(cells)}</div>" + "".join(parts))
    with open(os.path.join(out_dir, "groups.html"), "w", encoding="utf-8") as fh:
        fh.write(page)


def contact_sheet(members, crops_dir, out_path, cols=6, cell=(192, 192), limit=36):
    """Middle frames of up to ``limit`` crops, spread over the group."""
    if not members:
        return
    pick = [members[int(i)] for i in np.linspace(0, len(members) - 1, min(limit, len(members)))]
    tiles = []
    for n in pick:
        f = frames_at(os.path.join(crops_dir, n), 1)
        tile = np.zeros((cell[1], cell[0], 3), np.uint8)
        if f:
            img = f[0]
            s = min(cell[0] / img.shape[1], cell[1] / img.shape[0])
            img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)))
            y, x = (cell[1] - img.shape[0]) // 2, (cell[0] - img.shape[1]) // 2
            tile[y:y + img.shape[0], x:x + img.shape[1]] = img
        t = int(n.split("__")[1][:8]) // 1000
        cv2.putText(tile, f"{t // 60:02d}:{t % 60:02d}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)])
    cv2.imwrite(out_path, grid, [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    sys.exit(main())
