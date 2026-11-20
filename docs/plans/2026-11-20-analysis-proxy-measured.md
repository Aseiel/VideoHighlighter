# Analysing on a proxy: what it saves and what it costs (measured)

2026-11-20. One film, measured end to end, plus a 7-minute stretch of it
analysed both ways. The numbers here are what the app should quote when it
offers a proxy; they are not a promise for every file.

## The idea

Analysis runs on a smaller, frame-exact copy of the video (the *proxy*);
cuts and exports still come from the original. Editing software and action
cameras do the same: the camera writes a full-quality file and a small one
beside it.

**Frame-exact** is the whole point. The proxy keeps every frame with its own
timestamp and copies the audio untouched, so a moment found at 39:20 on the
proxy is 39:20 in the original, and every per-second result can be filed under
the original.

## The film

4320x2160 side-by-side VR, H.264, 60 fps, 48.8 minutes, 7.4 GB, 20 Mbit/s.

H.264 that wide cannot be decoded by the graphics cards' H.264 decoders (Intel
Arc and NVIDIA stop at 4096 on a side), so every pass decoded it on the
processor. OpenCV reported the card as decoding meanwhile; that is fixed
(`modules/media/video_decode.gpu_refusal`).

## Recipe

```
ffmpeg -i ORIGINAL -map 0:v:0 -map 0:a? -c:a copy
       -vf scale=2160:1080:flags=bicubic,format=nv12
       -c:v hevc_qsv -global_quality 24 -preset faster -g <fps>
       -fps_mode passthrough -map_metadata 0 PROXY
```

Half size each way, HEVC (which every card in the pool decodes in hardware,
the GTX 1060 included; AV1 would exclude it), a keyframe each second so the
machine pool can cut fine pieces. Scaling on the graphics card was tried and
fails on 4320-wide frames on both Arc machines, so it stays on the processor.

Checked: same frame count (175,783), same per-frame timestamps, mean pixel
difference 0.4 of 255 at 448x224. 1.13 GB, 6.5x smaller.

## Speed

| Step | Original | Proxy | Faster |
|---|---|---|---|
| Making the proxy (once) | — | 913 s on a Ryzen 5 5600; 1.9x faster on a Core Ultra 7 265K | |
| CLIP index, one PC | 830 s | 193 s | 4.3x |
| CLIP index, four cards on three PCs | 211 s | 54 s | 3.9x |
| Motion (every frame) | 2954 s | 718 s | 4.1x |
| Expressions (7-min stretch) | 329 s | 112 s | 2.9x |
| Face index (7-min stretch) | 142 s | 71 s | 2.0x |
| Audio, transcript, voices | same | same | the audio is copied |

Whole analysis through the proxy: about 53 minutes including making it. On
the original, motion alone took 49 minutes; the full set was stopped there and
is estimated at 2.5-3 hours.

## What changes in the results

| Result | Original vs proxy |
|---|---|
| CLIP frame vectors | cosine median 0.993 (lowest 0.977); 96.6 % of frames find their own moment within 1 s |
| Motion curve | correlation 0.9988, but **12 % lower** on average (less fine detail and noise) |
| Motion events / peaks over fixed thresholds | 94 % / 96 % of the original's found, but the counts differ (32,330 vs 20,131 events; 211 vs 771 peaks) |
| Scene changes | 14 of the original's 16 |
| Seconds with a face found | 5 % fewer (328 vs 345) |
| Expression label | same in 91 % of seconds; one label lost a fifth of its seconds |
| Face descriptions (CLIP), per second | correlation 0.75-0.87, 10-16 percentile points apart |
| Face descriptions, averaged over 30 s | correlation 0.89-0.99; gaze ("looking into the camera") weakest at 0.89 |

## Recommendation

- **Offer the proxy** when the cards cannot decode the original (H.264 above
  4096, or a codec they lack), or when the file is large enough that every
  pass would decode it on the processor. For a 1080p H.264 film the cards
  decode anyway and a proxy saves little.
- **Say what it costs** when offering it: on this film about 3-4x faster per
  pass, about 5 % fewer faces found, single-second face readings less exact;
  results over whole moments match closely.
- **Use the original where a single second matters**: faces (gaze above all)
  inside the few moments a report is about. A proxy for everything, the
  original for faces in short windows, keeps most of the speed.
- **Judge motion against the film, not a fixed number.** A rule like "motion
  above X" fires differently on a proxy; per-film percentiles do not care.
- **Make it on the fastest idle machine**, or split it across the pool like
  the CLIP index: decoding the original on the processor is the whole cost.
