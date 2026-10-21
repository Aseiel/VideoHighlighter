"""The language of the user's footage, taken from the interface language.

Choosing 简体中文 used to translate the menus and nothing else: Whisper still
listened for English (``transcript.source_lang: en``), subtitles were written
in Polish (the shipped ``target_lang``), narration came back in English, and
CLIP search read the query as English. A user with Chinese footage got bad
results with no hint why. So these settings now follow the interface language
until the user picks something else for them:

* ``transcript.source_lang`` (and its mirror ``subtitles.source_lang``): the
  language Whisper listens for;
* ``subtitles.target_lang``: the language subtitles are written in.

Narration is written in the interface language too, unless
``visualization.narration_language`` names another (``narration_instruction``);
it has no setting of its own to move.

"Until the user picks something else" is measured, not stored per field: when
the interface language changes, a setting moves along only if it still holds
what the *previous* interface language would have given it. A transcript set
to Japanese on purpose stays Japanese.

Configs written before this have ``source_lang: en`` and ``target_lang: pl``
saved whether anyone chose them or not, so ``migrate`` moves those once, the
first time a build with this module starts (marker: ``ui.content_follows_ui``).

Search follows on its own: llm/clip_multilingual.py reads the interface
language directly.

The first start: ``ui.language`` is left out of the shipped config, so a
config without it is one nobody has chosen a language for. The Windows
installer asks and leaves its answer in ``install_language.txt`` beside the
exe (``installer_choice``); everywhere else the app asks, preselecting the
system's language (``guess_from_locale``).
"""
from __future__ import annotations

import os
from typing import Optional

from modules.i18n.languages import DEFAULT_LANGUAGE, get, normalize_code, selectable_languages

# Interface language -> Whisper's code for it. Every shipped interface language
# is one Whisper transcribes.
WHISPER = {
    "en": "en", "zh-CN": "zh", "es": "es", "pt-BR": "pt", "fr": "fr", "ja": "ja",
    "de": "de", "ko": "ko", "id": "id", "tr": "tr", "it": "it", "pl": "pl",
    "vi": "vi", "nl": "nl",
}

# (section, key): the settings that follow. Each takes the Whisper code.
FOLLOWING = (
    ("transcript", "source_lang"),
    ("subtitles", "source_lang"),
    ("subtitles", "target_lang"),
)

# What builds before this one saved for those settings without being asked.
SHIPPED_DEFAULTS = {
    ("transcript", "source_lang"): "en",
    ("subtitles", "source_lang"): "en",
    ("subtitles", "target_lang"): "pl",
}

MIGRATED_KEY = "content_follows_ui"
INSTALL_FILE = "install_language.txt"


def whisper_code(ui_language: Optional[str]) -> str:
    return WHISPER.get(normalize_code(ui_language), "en")


def english_name(code: Optional[str]) -> str:
    """'Chinese (Simplified)' for 'zh-CN' or 'zh' alike; English for anything
    unknown. Narration prompts name the language in English, which every
    model reads."""
    for ui, whisper in WHISPER.items():
        if code in (ui, whisper):
            lang = get(ui)
            return lang.english_name if lang else "English"
    return "English"


def follow(config: dict, old_ui: Optional[str], new_ui: Optional[str],
           *, also_shipped: bool = False) -> list:
    """Move every following setting that still holds ``old_ui``'s value (or
    nothing) to ``new_ui``'s. With ``also_shipped``, the old builds' unasked
    defaults count as "nothing" too. Returns the (section, key) pairs changed.
    """
    old, new = whisper_code(old_ui), whisper_code(new_ui)
    changed = []
    for section, key in FOLLOWING:
        block = config.get(section)
        if block is None:
            block = config[section] = {}
        if not isinstance(block, dict):
            continue
        current = block.get(key)
        untouched = current in (None, "", old) or (
            also_shipped and current == SHIPPED_DEFAULTS.get((section, key)))
        if untouched and current != new:
            block[key] = new
            changed.append((section, key))
    return changed


def migrate(config: dict) -> list:
    """Once per config: line the following settings up with the interface
    language, counting the old builds' saved defaults as unchosen. Returns
    what changed; marks the config so it never runs again.

    An English interface changes nothing: the old defaults were English there
    already, and a ``target_lang`` someone kept at Polish may well be meant."""
    ui = config.setdefault("ui", {})
    if not isinstance(ui, dict) or ui.get(MIGRATED_KEY):
        return []
    changed = []
    if normalize_code(ui.get("language")) != DEFAULT_LANGUAGE:
        changed = follow(config, DEFAULT_LANGUAGE, ui.get("language"), also_shipped=True)
    ui[MIGRATED_KEY] = True
    return changed


def chosen(config: Optional[dict]) -> bool:
    """True when somebody has chosen the interface language in this config."""
    ui = (config or {}).get("ui") or {}
    return bool(str(ui.get("language") or "").strip())


def installer_choice(app_dir: str) -> Optional[str]:
    """The language the Windows installer asked for, or None."""
    path = os.path.join(app_dir, INSTALL_FILE)
    try:
        with open(path, encoding="utf-8-sig") as fh:
            raw = fh.read().strip()
    except OSError:
        return None
    code = normalize_code(raw)
    return code if (code != DEFAULT_LANGUAGE or raw == DEFAULT_LANGUAGE) else None


def guess_from_locale(tags) -> str:
    """The shipped interface language closest to the system's preferred ones
    ("zh-Hans-CN", "pl_PL", "pt-BR", ...), English when none is shipped."""
    if isinstance(tags, str):
        tags = [tags]
    shipped = {lang.code for lang in selectable_languages()}
    for tag in tags or ():
        tag = str(tag or "").replace("_", "-").strip()
        if not tag:
            continue
        low = tag.lower()
        if low.startswith("zh"):
            return "zh-CN" if "zh-CN" in shipped else DEFAULT_LANGUAGE
        if low.startswith("pt"):
            return "pt-BR" if "pt-BR" in shipped else DEFAULT_LANGUAGE
        base = low.split("-")[0]
        for code in shipped:
            if code.lower() == base:
                return code
    return DEFAULT_LANGUAGE


def narration_instruction(config: Optional[dict] = None) -> str:
    """A sentence for a narration system prompt naming the language to write
    in, or "" for English: ``visualization.narration_language`` when set,
    else the interface language (the running one when ``config`` is None)."""
    cfg = config or {}
    code = (cfg.get("visualization") or {}).get("narration_language")
    if not code:
        ui = (cfg.get("ui") or {}).get("language")
        if not ui:
            from modules.i18n.core import current_language
            ui = current_language()
        code = whisper_code(ui)
    if code in ("", "en", None):
        return ""
    return f"Write in {english_name(code)}, whatever language this prompt is in."
