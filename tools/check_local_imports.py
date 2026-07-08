"""Static, AST-based audit of local (repo-internal) Python imports.

Verifies that every local import in the repository resolves to a
git-tracked file, and that any named symbol it imports is actually
defined there. This is the safeguard for the bug class fixed in commit
0150a27: a file that exists on the author's machine but was never
`git add`ed, so a fresh clone breaks on import.

Never imports the target modules it checks -- everything here is static
`ast` parsing, so this module has no dependency beyond the standard
library and works without the project's heavy ML install.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from dataclasses import dataclass
from importlib.machinery import PathFinder
from pathlib import Path


class GitError(RuntimeError):
    """git is unavailable, or the target directory isn't a git checkout."""


@dataclass(frozen=True)
class LocalImport:
    importing_file: str  # repo-relative, forward slashes
    lineno: int
    kind: str  # "module" (plain `import a.b.c`) or "from" (`from a.b import c, d`)
    package_path: str  # repo-relative slash path with no extension: the submodule
    #   chain for kind="module", or the package/module being imported-from for kind="from"
    symbols: tuple[str, ...] | None  # kind="from" only; None for wildcard or kind="module"
    is_wildcard: bool = False


@dataclass(frozen=True)
class Violation:
    kind: str  # "untracked-or-missing-file" | "undefined-symbol"
    importing_file: str
    lineno: int
    detail: str

    def __str__(self) -> str:
        return f"{self.importing_file}:{self.lineno} -- {self.kind} -- {self.detail}"


def _run_git_ls_files(repo_root: Path, pattern: str) -> list[str]:
    try:
        result = subprocess.run(
            ["git", "ls-files", pattern],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise GitError("git not found -- is git installed and on PATH?") from exc
    if result.returncode != 0:
        raise GitError(
            f"'git ls-files {pattern}' failed in {repo_root} "
            f"(not a git repository?): {result.stderr.strip()}"
        )
    return [line for line in result.stdout.splitlines() if line]


def get_tracked_py_files(repo_root: Path) -> set[str]:
    """Repo-relative paths (forward slashes) of every git-tracked .py file."""
    return set(_run_git_ls_files(repo_root, "*.py"))


def _is_third_party(name: str, search_path: list[str]) -> bool:
    """True if `name` resolves to an installed distribution somewhere on
    `search_path`. Uses PathFinder directly against a caller-filtered path
    list rather than mutating global sys.path/sys.modules, so this has no
    side effects on caller state and is not affected by whatever already
    imported this process's own package (which would otherwise self-collide
    when the checked repo differs from the real project root, e.g. under a
    temp-directory test)."""
    try:
        spec = PathFinder.find_spec(name, path=search_path)
    except (ImportError, ValueError):
        spec = None
    return spec is not None


def enumerate_local_roots(repo_root: Path) -> set[str]:
    """Top-level names at repo_root that are local (repo-internal) import
    roots: a `<name>.py` file, or a `<name>/` directory containing at
    least one `.py` file anywhere in its tree -- excluding any candidate
    that also resolves to an installed third-party distribution."""
    roots: set[str] = set()
    for entry in repo_root.iterdir():
        if entry.name.startswith("."):
            continue
        if entry.is_file() and entry.suffix == ".py":
            roots.add(entry.stem)
        elif entry.is_dir():
            if any(entry.rglob("*.py")):
                roots.add(entry.name)
    resolved_root = repo_root.resolve()
    search_path = [p for p in sys.path if p and Path(p).resolve() != resolved_root]
    return {name for name in roots if not _is_third_party(name, search_path)}


def _relative_package_path(importing_file: str, level: int, module: str | None) -> str:
    """Repo-relative directory/module path a relative import's dots resolve
    to -- the package itself (bare `from . import X`) when `module` is
    None, or that package's named submodule (`from .sub import X`)."""
    parts = importing_file.split("/")
    package_parts = parts[:-1]  # directory containing importing_file
    climb = level - 1
    if climb > 0:
        package_parts = (
            package_parts[: len(package_parts) - climb] if climb <= len(package_parts) else []
        )
    prefix = "/".join(package_parts)
    if module:
        return f"{prefix}/{module.replace('.', '/')}" if prefix else module.replace(".", "/")
    return prefix


def extract_local_imports(
    tree: ast.AST, importing_file: str, local_roots: set[str]
) -> list[LocalImport]:
    """Walk every Import/ImportFrom node in `tree` (regardless of nesting
    inside try/except/if -- control flow never hides a statement from
    ast.walk) and return the ones that target a local root."""
    results: list[LocalImport] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                first_seg = alias.name.split(".")[0]
                if first_seg in local_roots:
                    results.append(
                        LocalImport(
                            importing_file=importing_file,
                            lineno=node.lineno,
                            kind="module",
                            package_path=alias.name.replace(".", "/"),
                            symbols=None,
                        )
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if not node.module:
                    continue
                first_seg = node.module.split(".")[0]
                if first_seg not in local_roots:
                    continue
                package_path = node.module.replace(".", "/")
            else:
                package_path = _relative_package_path(importing_file, node.level, node.module)

            is_wildcard = any(alias.name == "*" for alias in node.names)
            if is_wildcard:
                results.append(
                    LocalImport(
                        importing_file=importing_file,
                        lineno=node.lineno,
                        kind="from",
                        package_path=package_path,
                        symbols=None,
                        is_wildcard=True,
                    )
                )
            else:
                symbols = tuple(alias.name for alias in node.names)
                results.append(
                    LocalImport(
                        importing_file=importing_file,
                        lineno=node.lineno,
                        kind="from",
                        package_path=package_path,
                        symbols=symbols,
                    )
                )
    return results


_SCOPE_TRANSPARENT = (ast.Try, ast.If, ast.While, ast.For, ast.AsyncFor, ast.With, ast.AsyncWith)


def _collect_top_level_names(stmts: list[ast.stmt]) -> set[str]:
    """Names bound at module level: def/class names, plain assignment
    targets, and names bound by import statements. Recurses into
    module-level control flow (try/if/while/for/with) since those don't
    introduce a new scope in Python, but never into a function/class body."""
    names: set[str] = set()
    for node in stmts:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    continue
                names.add(alias.asname or alias.name)
        elif isinstance(node, _SCOPE_TRANSPARENT):
            for field in ("body", "orelse", "finalbody"):
                names |= _collect_top_level_names(getattr(node, field, []) or [])
            for handler in getattr(node, "handlers", []) or []:
                names |= _collect_top_level_names(handler.body)
        # Function/class/lambda bodies are a different scope -- never recursed into.
    return names


def _file_candidates(path: str) -> tuple[str, str]:
    """The two file forms a repo-relative module/package path could take."""
    return f"{path}.py", f"{path}/__init__.py"


def resolve_and_verify(
    local_imports: list[LocalImport],
    tracked_files: set[str],
    repo_root: Path,
    parsed_trees: dict[str, ast.AST] | None = None,
) -> list[Violation]:
    """Resolve each local import and verify it against `tracked_files`
    (git-tracked-ness) and, for named symbols, the target's own top-level
    bindings. For `from X import Y`, Y may be either a submodule of X or a
    symbol defined in X itself -- both are tried, matching how Python's
    import system actually resolves `from package import name`.

    `parsed_trees` lets a caller that already parsed a tracked file (e.g.
    `run_check`, which parses every tracked file once to extract its
    imports) share that tree instead of this function re-parsing it from
    disk when the same file is also a symbol-import target."""
    violations: list[Violation] = []
    parsed_trees = parsed_trees or {}
    top_level_cache: dict[str, set[str] | None] = {}

    def top_level_names(tracked_path: str) -> set[str] | None:
        if tracked_path in top_level_cache:
            return top_level_cache[tracked_path]
        tree = parsed_trees.get(tracked_path)
        if tree is None:
            full = repo_root / tracked_path
            try:
                source = full.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=tracked_path)
            except (OSError, SyntaxError):
                top_level_cache[tracked_path] = None
                return None
        names = _collect_top_level_names(tree.body)
        top_level_cache[tracked_path] = names
        return names

    def resolved_file(path: str) -> str | None:
        as_module, as_package = _file_candidates(path)
        if as_module in tracked_files:
            return as_module
        if as_package in tracked_files:
            return as_package
        return None

    def has_namespace_contents(path: str) -> bool:
        prefix = path + "/"
        return any(t.startswith(prefix) for t in tracked_files)

    for imp in local_imports:
        if imp.kind == "module":
            if resolved_file(imp.package_path) is None:
                as_module, as_package = _file_candidates(imp.package_path)
                violations.append(
                    Violation(
                        kind="untracked-or-missing-file",
                        importing_file=imp.importing_file,
                        lineno=imp.lineno,
                        detail=f"target '{imp.package_path}' resolves to neither "
                        f"'{as_module}' nor '{as_package}' in the git-tracked file set",
                    )
                )
            continue

        # kind == "from"
        package_file = resolved_file(imp.package_path)

        if imp.is_wildcard:
            if package_file is None and not has_namespace_contents(imp.package_path):
                violations.append(
                    Violation(
                        kind="untracked-or-missing-file",
                        importing_file=imp.importing_file,
                        lineno=imp.lineno,
                        detail=f"wildcard-imported package '{imp.package_path}' has no "
                        "tracked package file and no tracked contents",
                    )
                )
            continue

        for symbol in imp.symbols or ():
            submodule_candidate = f"{imp.package_path}/{symbol}"
            if resolved_file(submodule_candidate) is not None:
                continue  # valid submodule import (e.g. `from modules import debug_console`)

            if package_file is not None:
                names = top_level_names(package_file)
                if names is None or symbol in names:
                    continue  # valid symbol import, or target unparseable (not this checker's failure mode)
                violations.append(
                    Violation(
                        kind="undefined-symbol",
                        importing_file=imp.importing_file,
                        lineno=imp.lineno,
                        detail=f"'{symbol}' is not defined at module scope in '{package_file}', "
                        f"and '{submodule_candidate}.py' is not a tracked submodule either",
                    )
                )
                continue

            violations.append(
                Violation(
                    kind="untracked-or-missing-file",
                    importing_file=imp.importing_file,
                    lineno=imp.lineno,
                    detail=f"neither submodule '{submodule_candidate}' nor a package file for "
                    f"'{imp.package_path}' was found in the git-tracked file set",
                )
            )

    return violations


def run_check(repo_root: Path) -> list[Violation]:
    """Compose the full pipeline: enumerate roots, extract every local
    import from every tracked .py file, resolve and verify each."""
    tracked_files = get_tracked_py_files(repo_root)
    local_roots = enumerate_local_roots(repo_root)

    all_imports: list[LocalImport] = []
    parsed_trees: dict[str, ast.AST] = {}
    for tracked_path in sorted(tracked_files):
        full = repo_root / tracked_path
        try:
            source = full.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=tracked_path)
        except (OSError, SyntaxError):
            continue
        parsed_trees[tracked_path] = tree
        all_imports.extend(extract_local_imports(tree, tracked_path, local_roots))

    return resolve_and_verify(all_imports, tracked_files, repo_root, parsed_trees)


def _resolve_repo_root() -> Path:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise GitError("git not found -- is git installed and on PATH?") from exc
    if result.returncode != 0:
        raise GitError(f"not a git repository: {result.stderr.strip()}")
    return Path(result.stdout.strip())


def main() -> int:
    try:
        repo_root = _resolve_repo_root()
        violations = run_check(repo_root)
    except GitError as exc:
        print(f"check_local_imports: {exc}")
        return 1

    if not violations:
        print("check_local_imports: no violations found.")
        return 0

    for violation in violations:
        print(str(violation))
    print(f"check_local_imports: {len(violations)} violation(s) found.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
