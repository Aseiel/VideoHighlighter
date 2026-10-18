"""Merge new translations into the locale files and regenerate catalog.py.

    python tools/i18n_merge.py i18n_work            # merge and write
    python tools/i18n_merge.py i18n_work --dry-run  # report only

`i18n_work/<lang>.json` is {English msgid: translation} for strings the
catalog lacks; `i18n_work/<lang>.review.json`, when present, is
{msgid: corrected translation} for entries already in it. Every entry is
checked with modules/i18n/checks.py first: one that drops a format field, an
HTML tag or a line break, or adds a lone "&", is not merged. Entries already in
the catalog that fail those checks and have no passing correction are reset
to the English, so the UI shows that rather than broken markup. Reset, not
removed: every language keeps the same set of strings
(tests/test_i18n.py), and the reset ones are easy to find later as entries
equal to their key.

modules/i18n/locales/<lang>.json is the source; catalog.py is regenerated
from all of them, keys sorted case-insensitively as before. tools/
i18n_misses.py finds what is missing in the first place.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOCALES = REPO / "modules" / "i18n" / "locales"
CATALOG = REPO / "modules" / "i18n" / "catalog.py"

sys.path.insert(0, str(REPO))
from modules.i18n.checks import problems  # noqa: E402


def _ordered(d: dict, like: list) -> dict:
    """`d` in the order the existing file uses (sorted, if it was)."""
    if like == sorted(like):
        return dict(sorted(d.items()))
    if like == sorted(like, key=str.lower):
        return dict(sorted(d.items(), key=lambda kv: kv[0].lower()))
    return {**{k: d[k] for k in like if k in d},
            **{k: d[k] for k in sorted(d) if k not in like}}


def merge_language(lang: str, work: Path, report: dict) -> dict:
    path = LOCALES / f"{lang}.json"
    table = json.loads(path.read_text(encoding="utf-8"))
    order = list(table)
    rep = report.setdefault(lang, {"added": 0, "corrected": 0, "reset": 0,
                                   "rejected": []})

    review = work / f"{lang}.review.json"
    if review.exists():
        for k, v in json.loads(review.read_text(encoding="utf-8")).items():
            if k in table and not problems(k, v) and v != table[k]:
                table[k] = v
                rep["corrected"] += 1
            elif k in table and problems(k, v):
                rep["rejected"].append((k, problems(k, v)))

    new = work / f"{lang}.json"
    if new.exists():
        for k, v in json.loads(new.read_text(encoding="utf-8")).items():
            p = problems(k, v)
            if p:
                rep["rejected"].append((k, p))
            elif k not in table:
                table[k] = v
                rep["added"] += 1

    for k in [k for k, v in table.items() if problems(k, v)]:
        table[k] = k
        rep["reset"] += 1
    return _ordered(table, order)


def write_catalog(tables: dict[str, dict[str, str]]) -> None:
    src = CATALOG.read_text(encoding="utf-8")
    head = src[:src.index("TRANSLATIONS")]
    out = [head, "TRANSLATIONS: dict[str, dict[str, str]] = {\n"]
    for lang, table in tables.items():
        out.append(f"    {lang!r}: {{\n")
        for k in sorted(table, key=str.lower):
            out.append(f"        {k!r}: {table[k]!r},\n")
        out.append("    },\n")
    out.append("}\n")
    CATALOG.write_text("".join(out), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("work", help="folder holding <lang>.json / <lang>.review.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    work = Path(args.work)

    from modules.i18n import catalog
    report: dict = {}
    tables = {lang: merge_language(lang, work, report)
              for lang in catalog.TRANSLATIONS}      # keeps the catalog's order

    for lang, r in report.items():
        print(f"{lang:6} +{r['added']:<4} corrected {r['corrected']:<4} "
              f"reset {r['reset']:<4} rejected {len(r['rejected'])}")
        for k, p in r["rejected"][:5]:
            print(f"         rejected {k[:60]!r}: {', '.join(p)}")
    if args.dry_run:
        return 0
    for lang, table in tables.items():
        (LOCALES / f"{lang}.json").write_text(
            json.dumps(table, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_catalog(tables)
    print("wrote locales and catalog.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
