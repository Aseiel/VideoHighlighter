"""Persistent counter of analyzed videos.

Stores a tiny JSON file next to the user-editable config (project root from
source, next to the exe when frozen — see modules.app_paths.user_data_dir), so
the count survives restarts and travels with a packaged install.
"""

import json
import os
import threading

from modules.app_paths import user_data_dir

STATS_FILENAME = "analysis_stats.json"

_lock = threading.Lock()


def stats_path() -> str:
    return os.path.join(user_data_dir(), STATS_FILENAME)


def _load() -> dict:
    try:
        with open(stats_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def get_analyzed_count() -> int:
    """Lifetime number of successfully analyzed videos."""
    try:
        return int(_load().get("analyzed_videos", 0))
    except (TypeError, ValueError):
        return 0


def increment_analyzed(n: int = 1) -> int:
    """Add ``n`` to the lifetime counter and return the new total.

    Never raises: a stats file that can't be written (read-only install) just
    means the counter stays session-local.
    """
    with _lock:
        data = _load()
        try:
            total = int(data.get("analyzed_videos", 0)) + int(n)
        except (TypeError, ValueError):
            total = int(n)
        data["analyzed_videos"] = total
        try:
            tmp = stats_path() + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, stats_path())
        except Exception:
            pass
        return total
