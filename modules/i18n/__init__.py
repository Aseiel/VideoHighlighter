"""UI language support for VideoHighlighter.

English source strings are msgids. Phase-1 catalogs cover the easy LTR
languages from docs/UI-LANGUAGES.md; missing keys fall back to English.
"""

from modules.i18n.core import current_language, load_from_config, set_language, t
from modules.i18n.languages import (
    DEFAULT_LANGUAGE,
    TOP_20,
    display_label,
    get,
    normalize_code,
    selectable_languages,
)
from modules.i18n.qt_hooks import install as install_qt_hooks

__all__ = [
    "DEFAULT_LANGUAGE",
    "TOP_20",
    "current_language",
    "display_label",
    "get",
    "install_qt_hooks",
    "load_from_config",
    "normalize_code",
    "selectable_languages",
    "set_language",
    "t",
]
