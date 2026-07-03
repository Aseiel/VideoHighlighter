# Contributing to VideoHighlighter

First off — thanks for wanting to contribute! VideoHighlighter is a solo-built project that's growing, and outside contributions genuinely help.

Before you open a PR, please read this doc. It's short.

## How to Contribute

1. **Bugs / small fixes** — open an issue or PR directly. No need to ask first.
2. **New features / larger changes** — please open an issue first to discuss the approach before writing code. This avoids wasted work if the feature doesn't fit the project's direction.
3. **Questions / discussion** — join the [Discord](your-invite-link) and ask in `#support` or `#dev`.

### Development setup

See the main [README](./README.md) for the "Building from Source" instructions (Python + FFmpeg + dependencies).

### Pull request guidelines

- Keep PRs focused — one feature/fix per PR is easier to review than a bundle of unrelated changes.
- Briefly describe *what* changed and *why* in the PR description.
- If your change affects the `final_segments` pipeline (live preview / edit timeline), please note which side (pre-CompositionEngine vs post-filter) it touches.

## Contributor License Agreement (CLA)

VideoHighlighter is currently licensed under AGPL, with a commercial/pro license also planned or available for use cases where AGPL terms don't fit.

To keep that dual-licensing model possible, **all contributors must agree to a CLA before their first contribution is merged.**

**In plain terms, the CLA means:**

- **You keep copyright** on your own contributions. Contributing doesn't sign your work away.
- **You grant the project maintainer (Aseiel) a license to use your contribution** under the project's open-source license (AGPL) *and* under any current or future commercial license offered for VideoHighlighter — including the Pro version.
- This is what allows the project to stay open-source under AGPL while also offering a commercially licensed version, without needing to track down and re-clear permissions from every past contributor individually.

This is standard practice for dual-licensed open-source projects (e.g. MongoDB, GitLab, Sentry use similar models) — it's not unusual or contributor-hostile, it's what makes sustainable dual-licensing possible in the first place.

**How it works in practice:** on your first pull request, you'll be asked to confirm agreement to the CLA (via comment or bot, TBD as the project sets this up formally). No separate paperwork needed beyond that confirmation.

If you have questions about what this means for a specific contribution, ask in Discord or open an issue before submitting — happy to clarify.

## Code of Conduct

Be respectful, be constructive, assume good faith. Standard open-source etiquette applies.

---

Questions? [Join the Discord](your-invite-link) — `#support` for help, `#dev` for contribution discussion.