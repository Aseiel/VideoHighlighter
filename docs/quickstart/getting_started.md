# Getting Started with VideoHighlighter

## Installation

### Quick install (recommended)
Download the installer from [Releases](https://github.com/Aseiel/VideoHighlighter/releases):
- **Windows:** `00-VideoHighlighter-Windows-Setup.exe`
- **macOS:** Drag `.dmg` into Applications, then clear quarantine once
- **Linux:** Use pip to install from source

### From source
```bash
pip install -r requirements.txt
python main.py
```

## Your first analysis

### Step 1: Drop a video
Drag or paste a video file onto the "Drop files" zone, or click "Browse" to open one.

### Step 2: Choose highlight length
- **Short** — about 1–2 minutes (default)
- **Medium** — about 4 minutes  
- **Longer** — about 7 minutes

### Step 3: Analyze
Click **"Analyze"** and wait for scoring to complete. The app will:
1. Detect scene cuts, motion, audio peaks, objects, actions
2. Generate transcript (optional)
3. Score moments and rank them
4. Create the highlight reel and separate clips

### Step 4: Review results
- **Open timeline** — see scored moments on a timeline, edit if needed
- **Open report** — read why each moment was kept (point breakdown, detections, transcript claims)

### Step 5: Export
Export the highlight reel or individual clips to your desired location.

## Default scoring explained

The app automatically scores moments based on these signals:

| Signal | Points | Description |
|--------|--------|-------------|
| Scene cuts | +1 each | Strong visual transitions |
| Object detection | +5 (default) | When you name objects, this detects them |
| Motion peaks | 0 (disabled by default) | Generic motion when no object matches |
| Audio peaks | 0 (disabled) | Crowd noise, reactions, loud events |

**To enable more signals**, edit [`config.yaml`](../../config.yaml):

```yaml
scoring:
  audio_peak_points: 1       # Enable audio peaks
  loudness_burst_points: 1   # Enable loud bursts
  motion_peak_points: 1      # Enable generic motion
```

## What the app detects automatically

- **Scene cuts** — Strong visual transitions between shots
- **Motion events** — Areas with significant movement (disabled by default)
- **Audio peaks** — Loud segments like crowd noise or reactions (optional)
- **Loudness bursts** — Sudden volume changes (optional)

## What you teach it

The app is "user-taught" — nothing is preloaded. You define:

1. **Objects** — what to detect in the frame (via Training panel)
2. **Actions** — what activities are happening (type any action, or train your own)
3. **Keywords** — transcript phrases to highlight (optional)

See [`docs/DETECTION-GUIDE.md`](../DETECTION-GUIDE.md) for more about each detector type.

## Tips for best results

### On older hardware
Increase `advanced.frame_skip` in config.yaml:
- `5` = standard speed (default)
- `10+` = faster but less precise

### Tuning detections
Edit [`config.yaml`](../../config.yaml):
- Increase `objects.confidence` from `30` to `50` for fewer, higher-confidence detections
- Add classes to `actions.interesting` or `objects.interesting` for custom detection
- Enable scoring signals you care about

### Understanding the report
The HTML report explains:
- Which signals scored each moment (point breakdown)
- What objects/actions were detected at that time
- Transcript claims and their confidence
- Moments that didn't make the cut (and why)

## Next steps

- **Customize scoring:** Edit [`config.yaml`](../../config.yaml)
- **Teach it what to detect:** Use Training panel for custom models
- **Advanced rules:** See [`docs/DETECTION-GUIDE.md`](../DETECTION-GUIDE.md#4-the-composition-engine)
- **Troubleshooting:** Read the [FAQ](../../FAQ/faq.md)

## Getting help

- **Questions?** [Discord #support](https://discord.gg/cUPJqPAMmm)
- **Bugs/features?** [GitHub Issues](https://github.com/Aseiel/VideoHighlighter/issues)
- **Detailed guide:** [`docs/DETECTION-GUIDE.md`](../DETECTION-GUIDE.md)

---

**Remember:** Nothing is uploaded. All analysis runs locally on your machine.