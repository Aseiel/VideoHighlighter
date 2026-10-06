# VideoHighlighter FAQ

## Getting Started

### What detector should I use?
- **New to the app?** Start with CLIP search (no training needed) to explore your footage, then add trained detectors for specific needs.
- **Finding objects?** Use YOLOX object detection (80 COCO classes by default, or train your own).
- **Understanding activities?** Use action recognition: type any action (Kinetics-700 names are suggested), or train your own.
- **Want complex rules?** Use the Composition Engine to combine detections.

### How do I teach it what to look for?
See [`docs/DETECTION-GUIDE.md`](../DETECTION-GUIDE.md#2-actions) and [`CUSTOM-MODEL-TRAINING.md`](../CUSTOM-MODEL-TRAINING.md). Provide example clips, train a model, export the `.onnx`, and import it via Training panel.

### Can I run this on CPU only?
Yes! The app works on any modern CPU. GPU acceleration is optional and speeds up processing but isn't required for basic functionality.

## Performance

### Is this slow on my machine?
Processing time depends on:
- **GPU:** NVIDIA/CUDA, AMD/DirectML, or Intel/OpenVINO all supported
- **frame_skip** (config.yaml): Higher values = faster but less precise detection
- **Model size:** YOLOX `n` is smallest/fastest; try `s`, `m`, or `l` for better accuracy

Try increasing `advanced.frame_skip` from `5` to `10` on CPU-only machines.

### Why does it take so long?
Detection involves:
- Running models (object detection, action recognition) on every frame/window
- Whisper transcripting if enabled
- Scoring moments against your thresholds

The installer includes FFmpeg and downloads models automatically on first run.

## Configuration

### How do I change what gets detected?
Edit [`config.yaml`](../config.yaml):

```yaml
scoring:
  scene_points: 1           # Enable scene cuts
  motion_peak_points: 0     # Disable unless you want motion peaks
  loudness_burst_points: 0  # Disable unless you want audio bursts

actions:
  interesting: ["goal", "pass"]  # Add action classes to detect

objects:
  interesting: ["player", "ball"]  # Add object classes
  confidence: 30                    # Lower = more detections, higher = fewer
```

### How do I tune the scoring?
Each signal contributes points toward keeping a moment. Higher total scores win (up to `max_duration` seconds). Enable signals you want by giving them positive point values.

Default scoring prioritizes scene cuts (`scene_points: 1`) and object detections (`object_points: 5`).

### What does `advanced.frame_skip` do?
Controls processing rate:
- `5` = process every 5th frame (default, balanced speed)
- `10+` = faster but less precise
- Lower values = more accurate but slower

Adjust based on your hardware.

## Troubleshooting

### App won't start / crashes immediately
Check the debug log:
- Windows: `%APPDATA%\VideoHighlighter\debug.log`
- macOS: `~/Library/Application Support/VideoHighlighter/debug.log`  
- Linux: `~/.local/share/VideoHighlighter/debug.log`

Common causes:
- FFmpeg not found (installer bundles it)
- GPU driver mismatch (see vendor docs)
- Missing Python dependencies (run installer instead of portable mode)

### Report won't load / shows blank page
Reports are HTML files with embedded thumbnails. Open them in Chrome, Firefox, or Edge — older browsers may struggle.

### Transcript not working
Whisper models (~1GB) download on first run. If you see "No module named openai_whisper", Python isn't installed properly.

### No GPU acceleration detected
- **Intel Arc:** See [`INTEL-GPU.md`](../INTEL-GPU.md)
- **AMD:** See [`AMD-GPU.md`](../AMD-GPU.md) (uses DirectML on Windows)
- **NVIDIA:** Usually works automatically; check CUDA toolkit is installed

### Video playback issues
The app uses FFmpeg for decoding. If you see errors:
- Windows installer bundles FFmpeg
- Portable mode: download `ffmpeg.exe` and place beside the executable
- macOS/Linux: `brew install ffmpeg` or `apt install ffmpeg`

## Advanced Usage

### How do I create a custom detection rule?
See [`docs/DETECTION-GUIDE.md`](../DETECTION-GUIDE.md#4-the-composition-engine). Rules combine existing detections:

```yaml
ball_inside_net: "ball inside net"
two_players_nearby: "count(player, 2)"
player_chasing_ball: "player followed by ball"
```

### Can I export just the highlights?
Yes! After analysis, use "Open timeline" → select moments → export to your folder. Individual clips are also available via "Export separate clips".

### What does the report contain?
Each run generates a self-contained HTML report with:
- Per-moment point breakdown (what scored it)
- Object/action detections that fired
- Transcript claims with attribution
- Moments that scored well but didn't make the cut
- Settings used for this run

See [`docs/REPORTS.md`](../REPORTS.md) for details.

## Licensing

### Is this really free?
Yes, AGPLv3 license — free to use, modify, and redistribute. Commercial ("Pro") license also available from the copyright holders for specific use cases.

See [`LICENSE`](../../LICENSE) and [`CLA.md`](../CLA.md) for full terms.

## Contributing

### Can I contribute?
Absolutely! See [`CONTRIBUTING.md`](../CONTRIBUTING.md). The project welcomes:
- Bug reports (open an issue)
- Small fixes (open a PR)
- Feature proposals (discuss on Discord or open an issue first)

All contributors must agree to the CLA before their first PR is merged.

## Community

### Where do I ask questions?
- **General help:** [Discord #support](https://discord.gg/cUPJqPAMmm)
- **Development discussion:** [Discord #dev](https://discord.gg/cUPJqPAMmm)
- **Bug reports/ideas:** [GitHub Issues](../../issues)

### Can I share my own models?
Yes! See [`COMMUNITY-MODELS.md`](../COMMUNITY-MODELS.md). Requirements:
- Permissive license only (not AGPLv3)
- YOLOX-compatible format
- README.md included in package