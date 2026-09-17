"""Generate the Hugging Face model card (README.md) from a manifest."""
from __future__ import annotations

from .manifest import COMPLIANCE_ITEMS, METRIC_KEYS, TASKS, Manifest

APP_URL = "https://github.com/Aseiel/VideoHighlighter"
SITE_URL = "https://aseiel.github.io/VideoHighlighter-site/models.html"


def render_model_card(m: Manifest) -> str:
    task_name = TASKS[m.task][0]
    tags = "\n".join(f"- {t}" for t in m.hub_tags())
    labels = ", ".join(f"`{x}`" for x in m.labels[:50])
    if len(m.labels) > 50:
        labels += f" … ({len(m.labels)} total)"
    checklist = "\n".join(
        f"- [{'x' if m.compliance.get(k) else ' '}] {text}" for k, text in COMPLIANCE_ITEMS.items())
    subject = ""
    if m.game:
        subject = (f"\n**Made for:** {m.game} (community model, not affiliated with "
                   f"or endorsed by the game's publisher)\n")
    elif m.content_type:
        subject = f"\n**Made for:** {m.content_type}\n"

    measured = ""
    found = m.found_sentence()
    extra = [f"{METRIC_KEYS[k]}: {int(v) if float(v).is_integer() else v}"
             for k, v in m.metrics.items() if k not in ("heldout_found", "heldout_expected")]
    if found or extra:
        measured = "\n## How well it works\n\n"
        if found:
            measured += (f"{found}, measured by VideoHighlighter when the model was trained. "
                         "A model travels best to footage like the footage it learned from.\n")
        if extra:
            measured += "\n" + "\n".join(f"- {line}" for line in extra) + "\n"

    return f"""---
license: {m.license}
library_name: onnx
tags:
{tags}
---

# {m.display_name}

{m.description}
{subject}
| | |
|---|---|
| Task | {task_name} |
| Category | `{m.category}` |
| Labels | {labels} |
| Input | {m.input.width}×{m.input.height}, {m.input.color}, {m.input.layout}{f", {m.input.frames} frames" if m.task == "action_recognition" else ""} |
| Default confidence | {m.confidence_threshold} |
| Model version | {m.version} |
| Requires | VideoHighlighter {m.min_app_version} or newer |
| Author | {m.author} |
{measured}
## Use it in VideoHighlighter

In the app, open **Advanced → Community models…**, search for `{m.name}` and
click **Install**. VideoHighlighter checks the file and its checksum before using
it; the model then appears in the object model list. Browse all community
models at {SITE_URL}.

This is a community model for [VideoHighlighter]({APP_URL}). It runs locally;
no video leaves your computer.

## Training data and publishing checklist

This package contains **only model weights and metadata**. No clips, screenshots,
audio or other training data are included or distributed.

The author confirmed:

{checklist}

## Reporting a problem

If you are a rights holder and believe this model infringes your rights, open a
discussion on this model page or contact the VideoHighlighter maintainers via
{APP_URL}/issues. We will review and remove models where appropriate.

<!-- manifest {m.name} {m.version} {m.sha256.get(m.model_file, "")} -->
"""
