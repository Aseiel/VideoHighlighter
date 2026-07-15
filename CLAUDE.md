# VideoHighlighter

Python/PySide6 desktop app for finding and cutting highlights out of video.
This is the public AGPL edition. See `CONTRIBUTING.md` and `CLA.md` before
opening a PR.

## Content neutrality

Detection features here are **user-taught and content-neutral**: the mechanism
matches whatever the user gives it examples of, and it holds no opinion about
what that is. The user's categories are their own data and are defined at
runtime.

So, in the repo — source, comments, docstrings, tests, fixtures, label files,
preset lists, commit messages, UI strings:

- No NSFW/adult terminology, and no built-in prompt sets, presets, or category
  names for that content.
- Keep naming descriptive of the *mechanism* ("custom category", "prototype",
  "example frames"), never of any particular subject matter.

If a feature seems to require naming that content in the repo, the design is
wrong: make it user-supplied.

## Conventions

- The packaged exe is `--windowed`: `stdout` goes nowhere, so `modules/debug_console.py`
  tees all output to `debug.log` and the optional "Debug log" window. Diagnostic
  output belongs in `print()` (→ debug log); `append_log()` is the user-facing
  log pane and is only for things the user acts on.
- Dependencies should be permissive (MIT/BSD/Apache) where practical — prefer
  what is already in the stack over adding something new.
- Commit messages in this repo carry no `Co-Authored-By` trailer.
- Don't commit or push unless asked.
