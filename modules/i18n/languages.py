"""UI language registry — top-20 demand list and what phase 1 ships."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    code: str
    english_name: str
    native_name: str
    phase: int  # 1 = shipped UI catalog; 2+ = listed only for later
    # True when the combo should offer this language today.
    selectable: bool


# Full top-20 (see docs/UI-LANGUAGES.md). phase/selectable control the picker.
TOP_20: tuple[Language, ...] = (
    Language("en", "English", "English", 1, True),
    Language("zh-CN", "Chinese (Simplified)", "简体中文", 1, True),
    Language("es", "Spanish", "Español", 1, True),
    Language("hi", "Hindi", "हिन्दी", 2, False),
    Language("ar", "Arabic", "العربية", 3, False),
    Language("pt-BR", "Portuguese (Brazil)", "Português (Brasil)", 1, True),
    Language("fr", "French", "Français", 1, True),
    Language("ja", "Japanese", "日本語", 1, True),
    Language("de", "German", "Deutsch", 1, True),
    Language("ko", "Korean", "한국어", 1, True),
    Language("ru", "Russian", "Русский", 2, False),
    Language("id", "Indonesian", "Bahasa Indonesia", 1, True),
    Language("tr", "Turkish", "Türkçe", 1, True),
    Language("it", "Italian", "Italiano", 1, True),
    Language("pl", "Polish", "Polski", 1, True),
    Language("zh-TW", "Chinese (Traditional)", "繁體中文", 2, False),
    Language("vi", "Vietnamese", "Tiếng Việt", 1, True),
    Language("nl", "Dutch", "Nederlands", 1, True),
    Language("th", "Thai", "ไทย", 2, False),
    Language("uk", "Ukrainian", "Українська", 2, False),
)

DEFAULT_LANGUAGE = "en"

_BY_CODE = {lang.code: lang for lang in TOP_20}


def get(code: str) -> Language | None:
    return _BY_CODE.get((code or "").strip())


def selectable_languages() -> list[Language]:
    return [lang for lang in TOP_20 if lang.selectable]


def normalize_code(code: str | None) -> str:
    """Map a config value to a known code; unknown → English."""
    raw = (code or "").strip()
    if not raw:
        return DEFAULT_LANGUAGE
    if raw in _BY_CODE:
        return raw
    # Accept underscore form from some OS locales.
    alt = raw.replace("_", "-")
    if alt in _BY_CODE:
        return alt
    if raw.lower() == "pt-br" or alt.lower() == "pt-br":
        return "pt-BR"
    return DEFAULT_LANGUAGE


def display_label(lang: Language) -> str:
    if lang.code == "en":
        return lang.english_name
    return f"{lang.native_name} — {lang.english_name}"
