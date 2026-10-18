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

language with the same catalog as phase 1.



Runtime: `modules/i18n/` (same module in Free and Pro). Setting: `ui.language`

in `config.yaml` (About → Interface language). Default is `en`; a chosen

language is remembered until changed.



**Coverage model:** `i18n.install_qt_hooks()` patches Qt widgets so labels,
buttons, tabs, tooltips, placeholders, common message boxes, window and dock
titles, menus and actions, the status bar, and text drawn into a graphics
scene go through `t()`. Text painted by hand (the timeline's row names) calls
`i18n.t()` where it is drawn. Padding is not part of a msgid: `"  Audio"` is
looked up as `"Audio"`. An empty catalog entry falls back to English. The
catalog must contain every English UI string — English fallback is a catalog
gap to fix. Standalone timeline also loads `ui.language` and installs hooks.

Not translated yet: text built at runtime from data (`f"Loaded: {name}"`,
durations, most status-bar messages, the assistant's log lines) and spin-box
suffixes. Those need their call sites changed to translate a fixed template.

**Finding gaps:** `python tools/i18n_misses.py zh-CN --json out.json` lists
what a language lacks, from two passes: a scan for literals passed to hooked
Qt calls, and a run that builds the main and timeline windows in that
language and records every fallback. Neither alone is enough — strings that
reach Qt through a helper or a table are only seen at runtime, and context
menus and message boxes only by the scan.

**Adding translations:** put `<lang>.json` ({msgid: translation}) and,
optionally, `<lang>.review.json` (corrections to existing entries) in a work
folder, then `python tools/i18n_merge.py <folder>`. It checks every entry with
`modules/i18n/checks.py` (format fields, HTML tags, line breaks and a lone
`&` must match the English), writes `locales/<lang>.json`, and regenerates
`catalog.py` from them. An existing entry that fails the checks is reset to
the English. `tests/test_i18n_coverage.py` keeps the catalog free of such
entries and every language on the same set of strings.



## Later phases



- CJK remainder: `zh-TW`

- More LTR: `ru`, `uk`, `hi`

- RTL last: `ar` (layout mirroring)

