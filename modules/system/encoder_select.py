"""
modules/system/encoder_select.py
=========================
Pick the fastest working video encoder (+ ffmpeg args) for re-encoding
highlight clips. Shared by the editor (signal_timeline_viewer) and the
pipeline (video_cutter + final concat) so the codec decision lives in one
place.

Decision inputs:
  - GPU vendor from modules.system.device_utils.detect_best_device() — the same
    detection the pipeline already uses — so we try the machine's real
    hardware first instead of probing absent encoders.
  - Source resolution → H.264 (≤4096px) vs HEVC (VR / >4096px), because the
    hardware H.264 encoders top out at 4096px while VR frames are wider.
  - `ffmpeg -encoders` → only offer encoders this ffmpeg build actually has.

The chain always ends with CPU libx264 as a universal fallback. Callers try
it in order and fall through when an encoder fails at runtime (e.g. an
absent GPU, or an H.264 hardware encoder hitting the 4096px limit).
"""
import os
import subprocess

from modules.system.app_paths import ffmpeg_exe

_LIBX264 = ("libx264", ["-c:v", "libx264", "-preset", "fast", "-crf", "18"])
# CPU HEVC for high-res/VR. libx265 output matches how these VR sources are
# authored (x265, Main profile, hev1), which VR players like HereSphere accept
# — unlike the hardware HEVC encoders, whose bitstreams they reject. Slow, but
# the price of guaranteed VR playback.
_LIBX265 = ("libx265", ["-c:v", "libx265", "-preset", "fast", "-crf", "18",
                        "-tag:v", "hev1"])

_UNSET = object()
_vendor_cache = _UNSET
_encoders_cache = None
_size_cache = {}
_fps_cache = {}
_chain_cache = {}
_vr_cache = {}

RENDER_MODES = ("auto", "cpu", "gpu")

# HEVC levels (ITU-T H.265 table A.8): (level, max luma picture size, max luma
# samples per second). The level is written as the encoder's own number for it
# (6.1 -> 61), which is what hevc_qsv takes.
_HEVC_LEVELS = (
    (40, 2_228_224, 66_846_720),
    (41, 2_228_224, 133_693_440),
    (50, 8_912_896, 267_386_880),
    (51, 8_912_896, 534_773_760),
    (52, 8_912_896, 1_069_547_520),
    (60, 35_651_584, 1_069_547_520),
    (61, 35_651_584, 2_139_095_040),
    (62, 35_651_584, 4_278_190_080),
)


def hevc_level(width, height, fps):
    """The lowest HEVC level whose limits hold a `width`x`height` picture at
    `fps`, as the encoder's number (6.1 -> 61). 62 when nothing fits, 52 when
    the size is unknown.

    hevc_qsv needs it because its tier option only takes effect alongside an
    explicit level (see _hevc_qsv_args)."""
    if not (width and height):
        return 52
    luma = int(width) * int(height)
    rate = luma * float(fps or 60.0)
    side = max(int(width), int(height))
    for level, max_ps, max_sr in _HEVC_LEVELS:
        # A.4.1: neither side may exceed sqrt(8 * MaxLumaPs).
        if luma <= max_ps and rate <= max_sr and side * side <= 8 * max_ps:
            return level
    return 62


def _hevc_qsv_args(level):
    # Main tier, stated explicitly. ffmpeg's hevc_qsv defaults to High tier,
    # which neither the VR sources nor libx265 use; it is the one difference in
    # the stream's profile/tier/level header between QSV output and the files
    # headset players (HereSphere) accept, and a decoder that supports only
    # Main tier refuses a High-tier stream outright. QSV ignores -tier unless
    # -level is set too.
    return ["-c:v", "hevc_qsv", "-preset", "faster", "-global_quality", "22",
            "-tier", "main", "-level", str(level), "-tag:v", "hvc1"]


def _vendor_from_name(name):
    """'nvidia' | 'intel' | 'amd' | None from any string naming a GPU."""
    name = (name or "").lower()
    if "nvidia" in name or "geforce" in name or "cuda" in name or "rtx" in name:
        return "nvidia"
    if "intel" in name or "arc(tm)" in name or " arc " in name or "xpu" in name:
        return "intel"
    if "amd" in name or "radeon" in name:
        return "amd"
    return None


def preferred_gpu_vendor():
    """'nvidia' | 'intel' | 'amd' | None — the machine's GPU vendor from
    modules.system.device_utils.detect_best_device(). Cached. Returns None on
    CPU-only or if detection fails, and the caller then keeps the full
    candidate list.

    AMD is answered by the DirectML adapter name, not by the backend label.
    The label is a constant ("DirectML (AMD/DX12)") because DirectML runs on
    any DX12 card, so reading the vendor off it would pick AMF encoders on an
    NVIDIA machine that had DirectML forced on for testing — losing nvenc for
    an unrelated reason. The adapter name is the actual card.

    An AMD box without torch-directml installed still reports None, exactly as
    before: no detection, full candidate list, h264_amf still gets its turn.
    """

    global _vendor_cache
    if _vendor_cache is not _UNSET:
        return _vendor_cache
    vendor = None
    try:
        from modules.system.device_utils import detect_best_device
        info = detect_best_device(log_fn=lambda *a, **k: None)
        name = (getattr(info, "backend_name", "") or "").lower()
        if "cuda" in name or "nvidia" in name:
            vendor = "nvidia"
        elif "intel" in name or "xpu" in name:
            vendor = "intel"
        elif getattr(info, "dml_device", None):
            from modules.system import directml_device
            for adapter in directml_device.adapter_names():
                vendor = _vendor_from_name(adapter)
                if vendor:
                    break
    except Exception as e:
        print(f"⚠️ [encoder_select] device probe failed: {e}")
    _vendor_cache = vendor
    return vendor


def _available_encoders(ffmpeg=None):
    """Text of `ffmpeg -encoders` (stdout+stderr), cached."""
    global _encoders_cache
    if _encoders_cache is not None:
        return _encoders_cache
    text = ""
    try:
        out = subprocess.run([ffmpeg or ffmpeg_exe(), "-hide_banner", "-encoders"],
                             capture_output=True, text=True, timeout=15)
        text = (out.stdout or "") + (out.stderr or "")
    except Exception as e:
        print(f"⚠️ [encoder_select] could not probe ffmpeg encoders: {e}")
    _encoders_cache = text
    return text


def probe_video_size(video_path, ffmpeg=None):
    """(width, height) of the source, cached per path. Tries ffprobe, then
    cv2; returns (0, 0) if neither works (caller then assumes normal H.264)."""
    if video_path in _size_cache:
        return _size_cache[video_path]
    w = h = 0
    try:
        fp = ffmpeg or ffmpeg_exe()
        base = os.path.basename(fp)
        probe = fp
        if "ffmpeg" in base.lower():
            cand = os.path.join(os.path.dirname(fp),
                                base.lower().replace("ffmpeg", "ffprobe"))
            probe = cand if os.path.exists(cand) else "ffprobe"
        out = subprocess.run(
            [probe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "csv=p=0:s=x", video_path],
            capture_output=True, text=True, timeout=15)
        # Parse the first non-empty line only. Some files (e.g. GoPro HEVC) make
        # ffprobe emit the entry more than once, plus a blank line, even under
        # -select_streams v:0. Splitting the whole blob on "x" then yields
        # ['5312', '2988\r\n\r\n5312', '2988'] and int() blows up on element 1.
        for line in (out.stdout or "").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("x")
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                w, h = int(parts[0]), int(parts[1])
                break
    except Exception as e:
        print(f"⚠️ [encoder_select] ffprobe size probe failed: {e}")
    if not (w and h):
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
        except Exception:
            pass
    _size_cache[video_path] = (w, h)
    return (w, h)


def probe_video_fps(video_path):
    """Frame rate from the container, cached per path; 0.0 if unreadable."""
    if video_path in _fps_cache:
        return _fps_cache[video_path]
    fps = 0.0
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        cap.release()
    except Exception:
        pass
    if not (0 < fps < 1000):
        fps = 0.0
    _fps_cache[video_path] = fps
    return fps


def is_vr(video_path):
    """True when `video_path` looks like VR footage: wider than 4096 px (where
    the HEVC path, and the headset players, start) or side-by-side stereo as
    modules.media.vr_detect sees it. Cached per path."""
    if video_path in _vr_cache:
        return _vr_cache[video_path]
    w, h = probe_video_size(video_path)
    vr, why = False, ""
    if max(w, h) > 4096:
        vr, why = True, f"{w}x{h} is wider than 4096 px"
    else:
        try:
            from modules.media import vr_detect
            layout = vr_detect.probe(video_path)
            vr, why = bool(layout.side_by_side), layout.reason
        except Exception as e:
            why = f"layout probe failed: {e}"
    print(f"[encoder_select] {os.path.basename(video_path)}: "
          f"{'VR' if vr else 'not VR'} ({why})")
    _vr_cache[video_path] = vr
    return vr


def resolve_mode(video_path, mode):
    """'cpu' or 'gpu' for one video. 'auto' is CPU for VR footage, which has
    to play in a headset, and the hardware encoder for everything else.
    Anything unknown is treated as 'auto'."""
    if mode in ("cpu", "gpu"):
        return mode
    return "cpu" if is_vr(video_path) else "gpu"


def encoder_chain(video_path, ffmpeg=None, mode="gpu"):
    """Ordered list of (name, [ffmpeg video args]) to try, hardware first,
    always ending with a CPU fallback. Cached per (video path, mode).

    `mode`:
      - "gpu" (default): hardware first (nvenc/qsv/amf), CPU libx264 fallback.
        Fast. Hardware HEVC is forced to Main tier, which is what VR players
        were rejecting in QSV's default output.
      - "cpu": CPU only — libx265 for high-res/VR (matches how VR sources are
        authored, so VR players accept it), libx264 otherwise. Slow but safe.
      - "auto": "cpu" for VR footage, "gpu" otherwise (see resolve_mode).

    HEVC is used for high-res/VR sources (>4096px, what VR players expect);
    H.264 otherwise. When the GPU vendor is known only that vendor's encoder
    is kept (+ libx264); when unknown (CPU-only, or an AMD box with no
    DirectML installed to name the card) the full candidate list is kept so
    h264_amf/hevc_amf still gets a chance."""
    mode = resolve_mode(video_path, mode)
    cache_key = (video_path, mode)
    if cache_key in _chain_cache:
        return _chain_cache[cache_key]
    w, h = probe_video_size(video_path, ffmpeg)
    hi_res = max(w, h) > 4096
    if mode == "cpu":
        chain = [_LIBX265] if hi_res else [_LIBX264]
        print(f"[encoder_select] {os.path.basename(video_path)} {w}x{h} "
              f"({'HEVC' if hi_res else 'H.264'}), mode=cpu -> "
              f"{', '.join(n for n, _ in chain)}")
        _chain_cache[cache_key] = chain
        return chain
    text = _available_encoders(ffmpeg)
    if hi_res:
        candidates = [
            ("hevc_nvenc", ["-c:v", "hevc_nvenc", "-preset", "p5", "-rc",
                            "vbr", "-cq", "22", "-b:v", "0", "-tier", "main",
                            "-tag:v", "hvc1"]),
            ("hevc_qsv", _hevc_qsv_args(
                hevc_level(w, h, probe_video_fps(video_path)))),
            ("hevc_amf", ["-c:v", "hevc_amf", "-quality", "balanced", "-rc",
                          "cqp", "-qp_i", "22", "-qp_p", "22",
                          "-profile_tier", "main", "-tag:v", "hvc1"]),
        ]
    else:
        candidates = [
            ("h264_nvenc", ["-c:v", "h264_nvenc", "-preset", "p5", "-rc",
                            "vbr", "-cq", "20", "-b:v", "0"]),
            ("h264_qsv", ["-c:v", "h264_qsv", "-preset", "faster",
                          "-global_quality", "20"]),
            ("h264_amf", ["-c:v", "h264_amf", "-quality", "balanced", "-rc",
                          "cqp", "-qp_i", "20", "-qp_p", "20"]),
        ]
    present = [(n, a) for (n, a) in candidates if n in text]
    vendor = preferred_gpu_vendor()
    if vendor:
        key = {"nvidia": "nvenc", "intel": "qsv", "amd": "amf"}[vendor]
        present = [it for it in present if key in it[0]]
    chain = present + [_LIBX264]
    print(f"[encoder_select] {os.path.basename(video_path)} {w}x{h} "
          f"({'HEVC' if hi_res else 'H.264'}), GPU={vendor or 'unknown'} -> "
          f"{', '.join(n for n, _ in chain)}")
    _chain_cache[cache_key] = chain
    return chain
