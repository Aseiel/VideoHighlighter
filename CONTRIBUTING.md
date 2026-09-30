## Contributor License Agreement (CLA)

VideoHighlighter is licensed under **AGPLv3**, with a separate **commercial ("Pro") license** for use cases where AGPL terms don't fit.

To keep that dual-licensing model possible, **all contributors must agree to the [CLA](./CLA.md) before their first contribution is merged.**

**In plain terms:**

- **You keep copyright** on your own contributions. Contributing doesn't sign your work away.
- **You grant the project's copyright holders — Przemysław Kreft and Meric Donmezer — a license to use your contribution** under the project's open-source license (AGPLv3) *and* under any current or future commercial license offered for VideoHighlighter, including the Pro version.
- This is what allows the project to stay open-source under AGPL while also offering a commercially licensed version, without needing to track down and re-clear permissions from every past contributor individually.

This is standard practice for dual-licensed open-source projects (e.g., MongoDB, GitLab, Sentry use similar models) — it's not unusual or contributor-hostile, it's what makes sustainable dual-licensing possible in the first place.

**How to work:** on your first pull request, the CLA-assistant bot posts a comment asking you to confirm agreement. You reply once with the sentence it gives you, and that records your agreement against your GitHub identity — covering that PR and all future contributions. No separate paperwork. The full terms are in [CLA.md](./CLA.md).

If you have questions about what this means for a specific contribution, ask in Discord or open an issue before submitting — happy to clarify.

## Code quality expectations

Before opening a PR, make sure:

1. **Tests pass**: Run `python -m pytest` from the repo root. The test suite (~3200 tests) covers core functionality.
2. **Type hints added**: Core modules in `modules/`, `video_ai_editor/`, and `main.py` use type annotations for better tooling support.
3. **No new dependencies**: Prefer existing stack items; add only when truly necessary and permissive-licensed (MIT/BSD/Apache).
4. **Logging, not print()**: Use `print()` for user-facing output (→ debug log) or `append_log()` for interactive messages. Avoid logging library for production use.
5. **Commit messages**: Be descriptive of what changed and why. No `Co-Authored-By` trailers; if amending from a pushed branch, clear them in the merge box.

## Code style

- Follow PEP 8 (black-style formatting is recommended)
- Import order: stdlib → third-party → local modules
- Use `|` for union types (`int | None`) and `list[str]` instead of `List[str]`
- Docstrings for public functions and classes (Google style or reStructuredText, consistent with project)

## Pull request guidelines

- Keep PRs focused — one feature/fix per PR is easier to review than a bundle of unrelated changes.
- Briefly describe *what* changed and *why* in the PR description.
- If your change affects the `final_segments` pipeline (live preview / edit timeline), note which side it touches (pre-CompositionEngine vs post-filter).

## Code of Conduct

Be respectful, be constructive, assume good faith. Standard open-source etiquette applies.

---

Questions? [Join the Discord](https://discord.gg/cUPJqPAMmm) — `#support` for help, `#dev` for contribution discussion.