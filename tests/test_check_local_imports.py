"""Unit and meta-tests for tools/check_local_imports.py.

Covers U1 (local-root detection + import extraction) and U2 (resolution +
violation detection) scenarios from the plan, plus U4's CLI exit-code
behavior. These tests exercise the checker's own logic against synthetic
fixtures -- they do not assert anything about the real repository tree
(that is tests/test_local_import_completeness.py's job).
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from tools.check_local_imports import (
    GitError,
    enumerate_local_roots,
    extract_local_imports,
    get_tracked_py_files,
    resolve_and_verify,
    run_check,
)


# ---------------------------------------------------------------------------
# U1: extract_local_imports
# ---------------------------------------------------------------------------

LOCAL_ROOTS = {"modules", "video_ai_editor", "llm", "pipeline"}


def _imports_from_source(source: str, importing_file: str = "app.py", roots=LOCAL_ROOTS):
    tree = ast.parse(source, filename=importing_file)
    return extract_local_imports(tree, importing_file, roots)


def test_absolute_dotted_import_recognized_as_local():
    imports = _imports_from_source("import modules.foo\n")
    assert len(imports) == 1
    assert imports[0].kind == "module"
    assert imports[0].package_path == "modules/foo"
    assert imports[0].symbols is None
    assert not imports[0].is_wildcard


def test_from_import_captures_multiple_symbols():
    imports = _imports_from_source("from modules import foo, bar\n")
    assert len(imports) == 1
    assert imports[0].kind == "from"
    assert imports[0].package_path == "modules"
    assert imports[0].symbols == ("foo", "bar")


def test_wildcard_import_sets_wildcard_flag_no_symbol_list():
    imports = _imports_from_source("from modules.foo import *\n")
    assert len(imports) == 1
    assert imports[0].is_wildcard is True
    assert imports[0].symbols is None


def test_relative_imports_resolve_against_importing_file_path():
    imports = _imports_from_source(
        "from . import foo\nfrom .bar import baz\n",
        importing_file="llm/llm_chat_widget.py",
    )
    assert len(imports) == 2
    # Bare `from . import foo`: package_path is the current package itself,
    # `foo` is a candidate submodule or symbol within it.
    assert imports[0].package_path == "llm"
    assert imports[0].symbols == ("foo",)
    # `from .bar import baz`: package_path is the named submodule `llm/bar`.
    assert imports[1].package_path == "llm/bar"
    assert imports[1].symbols == ("baz",)


def test_third_party_imports_excluded():
    imports = _imports_from_source("import numpy\nfrom os import path\n")
    assert imports == []


def test_root_level_sibling_module_import_recognized_as_local():
    # Mirrors main.py's `from pipeline import run_highlighter` -- a
    # root-level sibling module, not a subpackage.
    imports = _imports_from_source(
        "from pipeline import run_highlighter\n", importing_file="main.py"
    )
    assert len(imports) == 1
    assert imports[0].package_path == "pipeline"
    assert imports[0].symbols == ("run_highlighter",)


def test_try_except_guarded_import_extracted_identically_to_unguarded():
    # Mirrors action_recognition.py:28 -- a real, confirmed pattern.
    source = (
        "try:\n"
        "    from modules.device_utils import detect_best_device\n"
        "    TORCH_AVAILABLE = True\n"
        "except ImportError:\n"
        "    TORCH_AVAILABLE = False\n"
    )
    imports = _imports_from_source(source, importing_file="action_recognition.py")
    assert len(imports) == 1
    assert imports[0].package_path == "modules/device_utils"
    assert imports[0].symbols == ("detect_best_device",)


# ---------------------------------------------------------------------------
# U1: enumerate_local_roots
# ---------------------------------------------------------------------------


def _make_repo(tmp_path: Path, layout: dict[str, str]) -> Path:
    for rel_path, content in layout.items():
        full = tmp_path / rel_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8")
    return tmp_path


def _init_git_repo(repo: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t.com", "-c", "user.name=t", "commit", "-q", "-m", "init"],
        cwd=repo,
        check=True,
    )


def test_enumerate_local_roots_finds_root_files_and_packages(tmp_path):
    # Deliberately fictional names -- this repo's own real top-level names
    # (pipeline, modules, ...) are already on sys.path via how pytest
    # imports this very test module, which would make them collide with
    # themselves under the collision guard regardless of tmp_path's content.
    repo = _make_repo(
        tmp_path,
        {
            "widgetmod.py": "x = 1\n",
            "gadgets/foo.py": "x = 1\n",  # no __init__.py -- namespace package
            "sprockets/__init__.py": "",
            "sprockets/bar.py": "x = 1\n",
        },
    )
    roots = enumerate_local_roots(repo)
    assert roots == {"widgetmod", "gadgets", "sprockets"}


def test_enumerate_local_roots_excludes_dirs_with_no_py_files(tmp_path):
    repo = _make_repo(tmp_path, {"models/weights.bin": "binary\n"})
    roots = enumerate_local_roots(repo)
    assert roots == set()


def test_enumerate_local_roots_collision_guard_excludes_third_party_name(tmp_path):
    # `packaging` is a real, near-universally-installed third-party
    # distribution (a transitive dependency of setuptools/pip). A repo
    # directory named `packaging/` containing only an unrelated script
    # must NOT be treated as a local import root.
    pytest.importorskip("packaging", reason="packaging must be installed for this test")
    repo = _make_repo(
        tmp_path, {"packaging/pyinstaller-hooks/hook-optimum.py": "x = 1\n"}
    )
    roots = enumerate_local_roots(repo)
    assert "packaging" not in roots


def test_enumerate_local_roots_raises_clear_error_when_git_unavailable(tmp_path, monkeypatch):
    def fake_run(*args, **kwargs):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(GitError):
        get_tracked_py_files(tmp_path)


def test_get_tracked_py_files_raises_clear_error_when_not_a_git_repo(tmp_path):
    with pytest.raises(GitError):
        get_tracked_py_files(tmp_path)


# ---------------------------------------------------------------------------
# U2: resolve_and_verify
# ---------------------------------------------------------------------------


def _local_import(package_path, symbols=None, is_wildcard=False, importing_file="app.py", lineno=1):
    from tools.check_local_imports import LocalImport

    return LocalImport(
        importing_file=importing_file,
        lineno=lineno,
        kind="module" if symbols is None and not is_wildcard else "from",
        package_path=package_path,
        symbols=symbols,
        is_wildcard=is_wildcard,
    )


def test_resolve_and_verify_no_violation_when_target_tracked_and_exists(tmp_path):
    repo = _make_repo(tmp_path, {"modules/foo.py": "def bar():\n    pass\n"})
    tracked = {"modules/foo.py"}
    imp = _local_import("modules/foo", symbols=("bar",))
    violations = resolve_and_verify([imp], tracked, repo)
    assert violations == []


def test_untracked_or_missing_violation_when_file_absent_from_disk(tmp_path):
    repo = tmp_path
    tracked: set[str] = set()
    imp = _local_import("modules/video_regions")
    violations = resolve_and_verify([imp], tracked, repo)
    assert len(violations) == 1
    assert violations[0].kind == "untracked-or-missing-file"


def test_untracked_or_missing_violation_when_present_on_disk_but_not_tracked(tmp_path):
    # The actual 0150a27 mechanism: file exists locally but was never
    # `git add`ed, so it's absent from the tracked-file set.
    repo = _make_repo(tmp_path, {"modules/video_regions.py": "def f():\n    pass\n"})
    tracked: set[str] = set()  # deliberately does NOT include the file present on disk
    imp = _local_import("modules/video_regions")
    violations = resolve_and_verify([imp], tracked, repo)
    assert len(violations) == 1
    assert violations[0].kind == "untracked-or-missing-file"


def test_undefined_symbol_violation_mirrors_edition_bug(tmp_path):
    repo = _make_repo(tmp_path, {"version.py": "__version__ = '0.8.1'\n"})
    tracked = {"version.py"}
    imp = _local_import("version", symbols=("__edition__",))
    violations = resolve_and_verify([imp], tracked, repo)
    assert len(violations) == 1
    assert violations[0].kind == "undefined-symbol"
    assert "__edition__" in violations[0].detail


def test_symbol_resolves_via_submodule_reexport_in_init(tmp_path):
    # Mirrors llm/llm_chat_widget.py:39's
    # `from .llm_timeline_bridge import TimelineBridge` pattern.
    repo = _make_repo(
        tmp_path,
        {
            "pkg/__init__.py": "from .submodule import name\n",
            "pkg/submodule.py": "def name():\n    pass\n",
        },
    )
    tracked = {"pkg/__init__.py", "pkg/submodule.py"}
    imp = _local_import("pkg", symbols=("name",))
    violations = resolve_and_verify([imp], tracked, repo)
    assert violations == []


def test_symbol_resolves_via_bare_submodule_reexport(tmp_path):
    repo = _make_repo(
        tmp_path,
        {
            "pkg/__init__.py": "from . import submodule\n",
            "pkg/submodule.py": "x = 1\n",
        },
    )
    tracked = {"pkg/__init__.py", "pkg/submodule.py"}
    imp = _local_import("pkg", symbols=("submodule",))
    violations = resolve_and_verify([imp], tracked, repo)
    assert violations == []


def test_symbol_resolves_via_aliased_import(tmp_path):
    repo = _make_repo(tmp_path, {"target.py": "import os as y\n"})
    tracked = {"target.py"}
    imp = _local_import("target", symbols=("y",))
    violations = resolve_and_verify([imp], tracked, repo)
    assert violations == []


def test_wildcard_import_never_produces_undefined_symbol_but_checks_target(tmp_path):
    repo = _make_repo(tmp_path, {"modules/foo.py": "x = 1\n"})
    tracked = {"modules/foo.py"}
    imp = _local_import("modules/foo", is_wildcard=True)
    violations = resolve_and_verify([imp], tracked, repo)
    assert violations == []  # target exists and is tracked; no symbol check runs

    imp_missing = _local_import("modules/missing", is_wildcard=True)
    violations = resolve_and_verify([imp_missing], tracked, repo)
    assert len(violations) == 1
    assert violations[0].kind == "untracked-or-missing-file"


def test_two_hop_reexport_resolves_without_inspecting_second_file(tmp_path):
    # a/__init__.py does `from .b import name`; consumer does `from a import name`.
    # `name` is bound directly in a/__init__.py's own top-level bindings,
    # so this resolves without needing to look inside b.py at all.
    repo = _make_repo(
        tmp_path,
        {
            "a/__init__.py": "from .b import name\n",
            "a/b.py": "name = 42\n",
        },
    )
    tracked = {"a/__init__.py", "a/b.py"}
    imp = _local_import("a", symbols=("name",))
    violations = resolve_and_verify([imp], tracked, repo)
    assert violations == []


def test_conditionally_defined_top_level_name_resolves(tmp_path):
    # Mirrors llm/llm_module.py:34-37's
    # `try: import cv2; HAS_CV2 = True \n except ImportError: HAS_CV2 = False`
    # -- a name assigned only inside a module-level try/except is still a
    # real top-level binding.
    repo = _make_repo(
        tmp_path,
        {
            "llm/llm_module.py": (
                "try:\n"
                "    HAS_CV2 = True\n"
                "except ImportError:\n"
                "    HAS_CV2 = False\n"
            )
        },
    )
    tracked = {"llm/llm_module.py"}
    imp = _local_import("llm/llm_module", symbols=("HAS_CV2",))
    violations = resolve_and_verify([imp], tracked, repo)
    assert violations == []


# ---------------------------------------------------------------------------
# run_check integration (fixture-tree level, not the real repo)
# ---------------------------------------------------------------------------


def test_run_check_clean_fixture_tree_reports_no_violations(tmp_path):
    # Fictional root name (see collision-guard note above) so this test
    # genuinely proves the clean-import path, not a false pass caused by
    # `gadgets`/`modules` self-colliding with this project's own package.
    repo = _make_repo(
        tmp_path,
        {
            "app.py": "from gadgets.foo import bar\n",
            "gadgets/foo.py": "def bar():\n    pass\n",
        },
    )
    _init_git_repo(repo)
    assert run_check(repo) == []


def test_run_check_flags_untracked_target(tmp_path):
    # Fictional root name -- avoids self-colliding with this project's own
    # real `modules` package, which is on sys.path during this test run.
    repo = _make_repo(
        tmp_path,
        {
            "app.py": "from gadgets.missing_module import bar\n",
            "gadgets/__init__.py": "",  # registers `gadgets` as a local root
        },
    )
    _init_git_repo(repo)
    violations = run_check(repo)
    assert len(violations) == 1
    assert violations[0].kind == "untracked-or-missing-file"


# ---------------------------------------------------------------------------
# U4: CLI exit-code behavior
# ---------------------------------------------------------------------------


def _run_cli(repo: Path) -> subprocess.CompletedProcess:
    module_path = Path(__file__).resolve().parents[1] / "tools" / "check_local_imports.py"
    return subprocess.run(
        [sys.executable, str(module_path)],
        cwd=repo,
        capture_output=True,
        text=True,
    )


def test_cli_exits_zero_on_clean_tree(tmp_path):
    repo = _make_repo(
        tmp_path,
        {
            "app.py": "from gadgets.foo import bar\n",
            "gadgets/foo.py": "def bar():\n    pass\n",
        },
    )
    _init_git_repo(repo)
    result = _run_cli(repo)
    assert result.returncode == 0
    assert "no violations" in result.stdout.lower()


def test_cli_exits_one_on_violation(tmp_path):
    repo = _make_repo(
        tmp_path,
        {
            "app.py": "from gadgets.missing import bar\n",
            "gadgets/__init__.py": "",  # registers `gadgets` as a local root
        },
    )
    _init_git_repo(repo)
    result = _run_cli(repo)
    assert result.returncode == 1
    assert "untracked-or-missing-file" in result.stdout
