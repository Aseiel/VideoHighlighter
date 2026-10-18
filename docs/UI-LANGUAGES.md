# UI languages



Application interface language (menus, tabs, labels) — not Whisper / subtitle

ISO codes. Those stay under Transcript / Subtitle settings (`en`, `zh`, …).



## Top 20 (priority by global demand)



Saved 2026-10-17. Order ≈ how often users want desktop software in that language.



| # | Language | Code | Phase |

|---|----------|------|--------|

| 1 | English | `en` | **shipped** (source) |

| 2 | Chinese (Simplified) | `zh-CN` | **shipped** (CJK phase 2) |

| 3 | Spanish | `es` | **phase 1** |

| 4 | Hindi | `hi` | later |

| 5 | Arabic | `ar` | later (RTL) |

| 6 | Portuguese (Brazil) | `pt-BR` | **phase 1** |

| 7 | French | `fr` | **phase 1** |

| 8 | Japanese | `ja` | **shipped** (CJK phase 2) |

| 9 | German | `de` | **phase 1** |

| 10 | Korean | `ko` | **shipped** (CJK phase 2) |

| 11 | Russian | `ru` | later |

| 12 | Indonesian | `id` | **phase 1** |

| 13 | Turkish | `tr` | **phase 1** |

| 14 | Italian | `it` | **phase 1** |

| 15 | Polish | `pl` | **phase 1** |

| 16 | Chinese (Traditional) | `zh-TW` | later (CJK) |

| 17 | Vietnamese | `vi` | **phase 1** |

| 18 | Dutch | `nl` | **phase 1** |

| 19 | Thai | `th` | later |

| 20 | Ukrainian | `uk` | later |



## Shipped UI catalogs



**Phase 1 (easy / LTR Latin):** `en`, `es`, `pt-BR`, `fr`, `de`, `id`, `tr`,

`it`, `pl`, `vi`, `nl`



**Phase 2 (CJK):** `zh-CN`, `ja`, `ko` — selectable in About → Interface

language with the same 364-string catalog as phase 1.



Runtime: `modules/i18n/` (same module in Free and Pro). Setting: `ui.language`

in `config.yaml` (About → Interface language). Default is `en`; a chosen

language is remembered until changed.



**Coverage model:** `i18n.install_qt_hooks()` patches Qt widgets so labels,
buttons, tabs, tooltips, placeholders, and common message boxes go through
`t()`. The catalog must contain every English UI string — English fallback is
a catalog gap to fix. Scan covers main, Simple view, Timeline Viewer, teach /
training, LLM chat, model hub, and `video_ai_editor` dialogs (~790 keys).
Standalone timeline also loads `ui.language` and installs hooks.

Regenerate: `tools/_export_msgids.py` → `_rebuild_locales_from_msgids.py` →
`_build_locale_jsons.py` → `_patch_locale_identities.py` →
`_merge_locales_to_catalog.py`.



## Later phases



- CJK remainder: `zh-TW`

- More LTR: `ru`, `uk`, `hi`

- RTL last: `ar` (layout mirroring)

