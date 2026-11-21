"""A spawned helper must not re-run the entry script (modules/system/child_processes.py)."""
import os
import subprocess
import sys
import textwrap
import types

import multiprocessing.spawn as spawn

from modules.system.child_processes import keep_main_out_of_children

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_preparation_data_names_no_main(monkeypatch):
    fake = types.ModuleType("__main__")
    fake.__file__ = os.path.join(REPO, "main.py")
    fake.__spec__ = types.SimpleNamespace(name="main")
    monkeypatch.setitem(sys.modules, "__main__", fake)

    keep_main_out_of_children()

    data = spawn.get_preparation_data("child")
    assert "init_main_from_path" not in data
    assert "init_main_from_name" not in data


def test_spawned_child_does_not_rerun_the_script(tmp_path):
    script = tmp_path / "entry.py"
    script.write_text(textwrap.dedent(f"""
        import os, sys
        sys.path.insert(0, {REPO!r})
        print("ENTRY RAN", flush=True)
        if __name__ == "__main__":
            import multiprocessing
            from modules.system.child_processes import keep_main_out_of_children
            keep_main_out_of_children()
            p = multiprocessing.get_context("spawn").Process(target=os.getpid)
            p.start(); p.join()
            print("CHILD EXIT", p.exitcode, flush=True)
    """))
    out = subprocess.run([sys.executable, str(script)], capture_output=True,
                         text=True, timeout=60, cwd=str(tmp_path)).stdout
    assert "CHILD EXIT 0" in out
    assert out.count("ENTRY RAN") == 1
