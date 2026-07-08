"""Production regression-prevention assertion: no local import in this
repository targets a file that is missing or untracked, and no imported
symbol is undefined. This is the safeguard for the bug class fixed in
commit 0150a27 -- see docs/plans/2026-07-08-001-test-local-import-audit-plan.md.
"""

from __future__ import annotations

from pathlib import Path

from tools.check_local_imports import run_check

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_no_local_import_violations():
    violations = run_check(REPO_ROOT)
    if violations:
        detail = "\n".join(str(v) for v in violations)
        raise AssertionError(
            f"{len(violations)} local import violation(s) found:\n{detail}"
        )
