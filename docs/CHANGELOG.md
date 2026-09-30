# Changelog

All notable changes to VideoHighlighter.

## [Unreleased] - Improvements in this session

### Bug fixes
- Fixed chat ownership management when switching between Simple and Detailed views. The app now correctly moves the single `LLMChatWidget` between views without losing state or model connections.

### Documentation improvements
- Added comprehensive FAQ (`docs/FAQ/faq.md`) covering:
  - Detector selection guidance for new users
  - Performance tuning (CPU/GPU, frame_skip, model sizes)
  - Configuration options explained with examples
  - Common troubleshooting scenarios (debug logs, FFmpeg, GPU issues, transcripts)
  - Advanced usage patterns (composition rules, report contents, model sharing)
- Updated `config.yaml` with detailed comments for all scoring, actions, objects, keywords, transcript, subtitles, and advanced options.
- Added troubleshooting section to `docs/INSTALL.md` covering debug logs, FFmpeg setup, GPU issues, transcript errors, report loading problems, and performance tuning.
- Improved `CONTRIBUTING.md` with code quality expectations (tests pass, type hints, no new dependencies, logging patterns) and style guidelines (PEP 8, import order, union types, docstrings).
- Added FAQ link to main `README.md`.

### Developer experience
- Clarified chat panel transfer behavior between Simple and Detailed views. The single assistant instance survives view switches seamlessly.
