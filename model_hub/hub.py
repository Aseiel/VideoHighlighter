"""Hugging Face integration: publish, browse and install community models.

Hugging Face stores the files; the app and the website only list and link them.
Downloads are pinned to an exact commit and re-checked with package.check_package
before a model is installed.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .manifest import (
    HUB_TAG, MANIFEST_NAME, TASKS, USABLE_TASKS, Manifest, category_from_tags,
)
from .package import UPLOAD_PATTERNS, CheckReport, check_package, sha256_file

KEYRING_SERVICE = "VideoHighlighter"
KEYRING_USER = "huggingface"
# Models under these accounts were reviewed by the maintainers.
VERIFIED_AUTHORS = {"VideoHighlighter"}
REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")

# Repositories the maintainers delisted after a report. Published on the site
# as plain JSON, read by the app and by the site's own models page, so a
# delisting takes effect everywhere by editing one file.
BLOCKLIST_URL = "https://aseiel.github.io/VideoHighlighter-site/models-blocklist.json"

# Written into an install folder by the app itself, never downloaded.
INSTALL_RECORD = "install.json"
LABELS_SIDECAR = "labels.json"   # what modules.vision.detection_backend reads beside a model
INSTALLED_EXTRA = frozenset({INSTALL_RECORD, LABELS_SIDECAR})

# The kraken is what we call the moment a package is cleared to leave this
# machine. It is released once per upload, only after every check has passed.
KRAKEN = "🐙"

Progress = Callable[[str], None]


def default_models_dir() -> Path:
    """Where community models install: ``<app data>/models/community``.

    Beside ``models/custom`` (the user's own models), in the app's writable
    data directory, so a portable install keeps its models with it.
    ``VIDEOHIGHLIGHTER_HOME`` overrides the data directory.
    """
    base = os.environ.get("VIDEOHIGHLIGHTER_HOME")
    if base:
        return Path(base) / "models" / "community"
    try:
        from modules.system.app_paths import user_data_dir
        return Path(user_data_dir()) / "models" / "community"
    except Exception:  # noqa: BLE001 - used outside the app
        return Path.home() / ".videohighlighter" / "models" / "community"


# ------------------------------------------------------------------- tokens
def get_token() -> str | None:
    try:
        import keyring
        return keyring.get_password(KEYRING_SERVICE, KEYRING_USER)
    except Exception:  # noqa: BLE001 - no keyring backend available
        return None


def save_token(token: str) -> bool:
    """Store the token in the OS credential store. Returns False if unavailable."""
    try:
        import keyring
        keyring.set_password(KEYRING_SERVICE, KEYRING_USER, token.strip())
        return True
    except Exception:  # noqa: BLE001
        return False


def forget_token() -> None:
    try:
        import keyring
        keyring.delete_password(KEYRING_SERVICE, KEYRING_USER)
    except Exception:  # noqa: BLE001
        pass


# ------------------------------------------------------------------ publish
def release_the_kraken(package_dir: str | Path, progress: Progress = print) -> Manifest:
    """The last gate before a package leaves this machine. Returns its manifest.

    Re-runs every check a stranger's install would run — layout, manifest,
    checksum, one test inference — with the publishing checklist required.
    Only when all of that holds is the kraken released, and ``publish`` uploads
    nothing it did not clear.
    """
    report, manifest = check_package(package_dir, require_compliance=True)
    if not report.ok or manifest is None:
        raise RuntimeError("The package did not pass the checks:\n" + report.text())
    line = (f"{KRAKEN} Release the kraken: {manifest.display_name} "
            f"({manifest.category}, {manifest.license}) is cleared to share.")
    try:
        progress(line)
    except UnicodeEncodeError:
        # A legacy-codepage console cannot print the octopus. The kraken is
        # released all the same; a log line must never block a checked upload.
        progress(line.replace(KRAKEN, "(kraken)").encode("ascii", "replace").decode("ascii"))
    return manifest


def publish(package_dir: str | Path, repo_name: str | None = None, token: str | None = None,
            private: bool = False, progress: Progress = print) -> str:
    """Check the package and upload it to the user's Hugging Face account.

    Returns the model page URL. Raises RuntimeError with a readable message.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    package_dir = Path(package_dir)
    progress("Checking package…")
    manifest = release_the_kraken(package_dir, progress=progress)

    token = token or get_token()
    if not token:
        raise RuntimeError("Sign in with a Hugging Face access token (write permission) first.")

    api = HfApi(token=token)
    try:
        user = api.whoami()["name"]
    except HfHubHTTPError as exc:
        raise RuntimeError("Hugging Face rejected the token. Create a new token with "
                           "write permission at huggingface.co/settings/tokens.") from exc

    repo_id = f"{user}/{repo_name or manifest.name}"
    if not REPO_ID_RE.match(repo_id):
        raise RuntimeError(f"Invalid repository name: {repo_id}")

    try:
        progress(f"Creating {repo_id}…")
        api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
        progress("Uploading model files…")
        commit = api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(package_dir),
            allow_patterns=UPLOAD_PATTERNS,  # nothing else can leave the computer
            commit_message=f"Publish {manifest.name} {manifest.version} via VideoHighlighter",
        )
    except HfHubHTTPError as exc:
        raise RuntimeError(f"Upload failed: {exc}") from exc

    progress(f"Published (commit {commit.oid[:8]}).")
    return f"https://huggingface.co/{repo_id}"


# ------------------------------------------------------------------ blocklist
def parse_blocklist(data) -> set[str]:
    """Repository ids from ``{"blocked": [{"repo_id": ..., "reason": ...}, ...]}``
    (or a bare list of ids), lower-cased — Hugging Face ids are case-insensitive."""
    items = data.get("blocked", []) if isinstance(data, dict) else data
    ids = set()
    for item in items if isinstance(items, list) else []:
        repo = item.get("repo_id") if isinstance(item, dict) else item
        if isinstance(repo, str) and REPO_ID_RE.match(repo):
            ids.add(repo.lower())
    return ids


def fetch_blocklist(url: str = BLOCKLIST_URL, timeout: float = 10.0) -> set[str]:
    """The current blocklist, or an empty set when the site cannot be reached.

    Failing open is deliberate: a flaky connection must not hide every model.
    The cost is that a delisted model stays visible to someone offline from the
    site but online to Hugging Face — rare, and install is re-checked anyway.
    """
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return parse_blocklist(json.loads(resp.read(1_000_000).decode("utf-8")))
    except Exception as exc:  # noqa: BLE001
        print(f"[model_hub] blocklist unavailable ({exc}); listing without it")
        return set()


# ------------------------------------------------------------------ catalog
@dataclass
class CatalogEntry:
    repo_id: str
    author: str
    task: str
    category: str
    downloads: int
    likes: int
    last_modified: str
    sha: str
    tags: list[str]

    @property
    def verified(self) -> bool:
        return self.author in VERIFIED_AUTHORS

    @property
    def usable(self) -> bool:
        return self.task in USABLE_TASKS

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo_id}"


def _task_from_tags(tags: list[str]) -> str:
    for t in tags:
        if t.startswith("vh-") and t[3:].replace("-", "_") in TASKS:
            return t[3:].replace("-", "_")
    return ""


def search_catalog(query: str = "", task: str = "", category: str = "",
                   limit: int = 200) -> list[CatalogEntry]:
    """List public models tagged 'videohighlighter', verified first, then most
    downloaded. ``category`` keeps a category and everything under it."""
    from huggingface_hub import HfApi

    tags = [HUB_TAG] + ([f"vh-{task.replace('_', '-')}"] if task else [])
    models = HfApi().list_models(filter=tags, search=query or None, sort="downloads",
                                 limit=limit, full=True)
    blocked = fetch_blocklist()
    entries = []
    for m in models:
        if m.id.lower() in blocked:
            continue
        mtags = list(m.tags or [])
        entry = CatalogEntry(
            repo_id=m.id,
            author=m.author or m.id.split("/")[0],
            task=_task_from_tags(mtags),
            category=category_from_tags(mtags),
            downloads=int(m.downloads or 0),
            likes=int(m.likes or 0),
            last_modified=str(m.last_modified or ""),
            sha=m.sha or "",
            tags=mtags,
        )
        if category and not (entry.category == category or entry.category.startswith(category + "/")):
            continue
        entries.append(entry)
    entries.sort(key=lambda e: (not e.verified, -e.downloads))
    return entries


def categories_in(entries: list[CatalogEntry]) -> list[str]:
    """Every category (and parent category) the entries name, sorted — the
    vocabulary comes from what people published, never from code."""
    found = set()
    for e in entries:
        parts = e.category.split("/") if e.category else []
        for i in range(1, len(parts) + 1):
            found.add("/".join(parts[:i]))
    return sorted(found)


# ------------------------------------------------------------------ install
@dataclass
class InstalledModel:
    repo_id: str
    revision: str
    path: Path
    manifest: Manifest
    verified: bool

    @property
    def model_path(self) -> Path:
        return self.path / self.manifest.model_file


def _safe_dir_name(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def install(repo_id: str, revision: str | None = None, models_dir: str | Path | None = None,
            progress: Progress = print) -> tuple[InstalledModel | None, CheckReport]:
    """Download a model pinned to one commit, check it, then install it."""
    if not REPO_ID_RE.match(repo_id):
        report = CheckReport(errors=[f"Invalid repository id: {repo_id}"])
        return None, report
    if repo_id.lower() in fetch_blocklist():
        return None, CheckReport(errors=[
            f"{repo_id} was removed from the community models after a report."])

    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    info = api.model_info(repo_id, revision=revision)
    revision = info.sha  # pin the exact commit we are about to check
    models_dir = Path(models_dir) if models_dir else default_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=models_dir) as tmp:
        progress(f"Downloading {repo_id}@{revision[:8]}…")
        snapshot_download(repo_id, revision=revision, local_dir=tmp,
                          allow_patterns=UPLOAD_PATTERNS)
        cache = Path(tmp) / ".cache"
        if cache.exists():
            shutil.rmtree(cache)  # huggingface_hub bookkeeping, not part of the package
        progress("Checking model…")
        return install_folder(tmp, repo_id, revision, models_dir, progress)


def install_folder(package_dir: str | Path, repo_id: str, revision: str,
                   models_dir: str | Path | None = None,
                   progress: Progress = print) -> tuple[InstalledModel | None, CheckReport]:
    """Check a package folder and install it. The offline half of ``install``,
    also used for a package someone received as files."""
    models_dir = Path(models_dir) if models_dir else default_models_dir()
    report, manifest = check_package(package_dir, require_compliance=True)
    if not report.ok or manifest is None:
        return None, report
    if manifest.task not in USABLE_TASKS:
        report.errors.append(
            f"This is a {TASKS[manifest.task][0].lower()} model; this version of "
            "VideoHighlighter can only use object detection models.")
        return None, report

    target = models_dir / _safe_dir_name(repo_id) / revision
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(package_dir, target)

    author = repo_id.split("/")[0]
    record = {"repo_id": repo_id, "revision": revision, "verified": author in VERIFIED_AUTHORS}
    (target / INSTALL_RECORD).write_text(json.dumps(record, indent=2), encoding="utf-8")
    # The detector backend names classes from a labels.json beside the model.
    (target / LABELS_SIDECAR).write_text(json.dumps(manifest.labels, indent=2), encoding="utf-8")
    progress("Installed.")
    return InstalledModel(repo_id, revision, target, manifest, author in VERIFIED_AUTHORS), report


def list_installed(models_dir: str | Path | None = None) -> list[InstalledModel]:
    """Installed models (manifest read, nothing executed)."""
    models_dir = Path(models_dir) if models_dir else default_models_dir()
    result = []
    if not models_dir.is_dir():
        return result
    for rec_file in models_dir.glob(f"*/*/{INSTALL_RECORD}"):
        folder = rec_file.parent
        try:
            rec = json.loads(rec_file.read_text(encoding="utf-8"))
            manifest = Manifest.load(folder)
        except Exception:  # noqa: BLE001 - skip broken installs
            continue
        result.append(InstalledModel(rec["repo_id"], rec["revision"], folder, manifest,
                                     bool(rec.get("verified"))))
    return result


def verify_installed(model: InstalledModel) -> CheckReport:
    """Every check, including a test inference; catches files changed on disk."""
    report, _ = check_package(model.path, require_compliance=True,
                              extra_allowed=INSTALLED_EXTRA)
    return report


_CHECKSUM_CACHE: dict = {}


def checksum_ok(model: InstalledModel) -> bool:
    """Cheap re-check for listing: the model file still matches its manifest.
    Cached per (path, size, mtime), so a combo rebuild does not re-hash."""
    path = model.model_path
    try:
        st = path.stat()
    except OSError:
        return False
    key = (str(path), st.st_size, st.st_mtime)
    if key not in _CHECKSUM_CACHE:
        _CHECKSUM_CACHE[key] = sha256_file(path) == model.manifest.sha256.get(model.manifest.model_file)
    return _CHECKSUM_CACHE[key]


def installed_detectors(models_dir: str | Path | None = None) -> list[dict]:
    """Community detectors ready for the object model picker, in the shape
    ``modules.system.app_paths.discover_object_models`` returns:
    ``[{"path", "name", "classes", "community": repo_id}]``. A model whose file
    no longer matches its checksum is left out."""
    out = []
    for m in list_installed(models_dir):
        if m.manifest.task != "object_detection" or not checksum_ok(m):
            continue
        out.append({"path": str(m.model_path), "name": m.manifest.display_name,
                    "classes": list(m.manifest.labels), "community": m.repo_id})
    return out


def uninstall(model: InstalledModel) -> None:
    shutil.rmtree(model.path, ignore_errors=True)
    parent = model.path.parent
    if parent.is_dir() and not any(parent.iterdir()):
        parent.rmdir()


def read_remote_manifest(repo_id: str, revision: str | None = None) -> Manifest:
    """Fetch only videohighlighter.json, e.g. to show details before installing."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id, MANIFEST_NAME, revision=revision)
    return Manifest.load(Path(path).parent)
