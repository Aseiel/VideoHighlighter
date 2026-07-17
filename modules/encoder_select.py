"""
modules/encoder_select.py
=========================
Pick the fastest working video encoder (+ ffmpeg args) for re-encoding
highlight clips. Shared by the editor (signal_timeline_viewer) and the
pipeline (video_cutter + final concat) so the codec decision lives in one
place.

Decision inputs:
  - GPU vendor from modules.device_utils.detect_best_device() — the same
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

from modules.app_paths import ffmpeg_exe

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
_chain_cache = {}


def preferred_gpu_vendor():
    """'nvidia' | 'intel' | None — the machine's GPU vendor from
    modules.device_utils.detect_best_device(). Cached. Returns None on
    CPU-only or if detection fails (device_utils does not detect AMD, so an
    AMD box reports None and the caller keeps the full candidate list)."""
    global _vendor_cache
    if _vendor_cache is not _UNSET:
        return _vendor_cache
    vendor = None
    try:
        from modules.device_utils import detect_best_device
        info = detect_best_device(log_fn=lambda *a, **k: None)
        name = (getattr(info, "backend_name", "") or "").lower()
        if "cuda" in name or "nvidia" in name:
            vendor = "nvidia"
        elif "intel" in name or "xpu" in name:
            vendor = "intel"
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
        parts = (out.stdout or "").strip().split("x")
        if len(parts) >= 2:
            w, h = int(parts[0]), int(parts[1])
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


def encoder_chain(video_path, ffmpeg=None, mode="gpu"):
    """Ordered list of (name, [ffmpeg video args]) to try, hardware first,
    always ending with a CPU fallback. Cached per (video path, mode).

    `mode`:
      - "gpu" (default): hardware first (nvenc/qsv/amf), CPU libx264 fallback.
        Fast, but the hardware HEVC bitstreams are rejected by some VR players.
      - "cpu": CPU only — libx265 for high-res/VR (matches how VR sources are
        authored, so VR players accept it), libx264 otherwise. Slow but safe.

    HEVC is used for high-res/VR sources (>4096px, what VR players expect);
    H.264 otherwise. When the GPU vendor is known only that vendor's encoder
    is kept (+ libx264); when unknown (CPU-only / AMD) the full candidate list
    is kept so h264_amf/hevc_amf still gets a chance."""
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
                            "vbr", "-cq", "22", "-b:v", "0", "-tag:v", "hvc1"]),
            ("hevc_qsv", ["-c:v", "hevc_qsv", "-preset", "faster",
                          "-global_quality", "22", "-tag:v", "hvc1"]),
            ("hevc_amf", ["-c:v", "hevc_amf", "-quality", "balanced", "-rc",
                          "cqp", "-qp_i", "22", "-qp_p", "22", "-tag:v", "hvc1"]),
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
