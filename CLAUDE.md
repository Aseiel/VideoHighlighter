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

## Never rewrite this repo's history

This repo takes pull requests from outside the project, and a contributor's
commits are the only record that they were here. **`main` is append-only: push
fast-forward, never force.** Not to drop a trailer, not to tidy a message, not
to re-attribute a merge. If a push needs `--force`, stop and ask the maintainer
first — no reason clears that bar on its own.

A rewrite re-hashes every commit from the rewrite point forward, contributors'
included. GitHub's contributor index is cached against the *old* hashes and does
not follow, so an outside contributor silently disappears from the sidebar and
the graph — while people whose commits are long gone stay listed. The only
forced repair is a GitHub Support ticket.

- Never `filter-repo`, `rebase`, `commit --amend`, or `reset --hard` anything
  already pushed to `main`.
- A contributor's authorship is theirs. Don't "fix" it by re-authoring their
  commits or by adding `Co-Authored-By` to a merge commit — the author field
  already credits them, and a trailer is strictly weaker.
- If someone is missing from the contributors sidebar, that is almost always the
  stale cache rather than the repo. Check the commit's `author.login` via the
  API first. The answer is to wait for the recompute or to open a support
  ticket — never to push again harder.

## Conventions

- The packaged exe is `--windowed`: `stdout` goes nowhere, so `modules/debug_console.py`
  tees all output to `debug.log` and the optional "Debug log" window. Diagnostic
  output belongs in `print()` (→ debug log); `append_log()` is the user-facing
  log pane and is only for things the user acts on.
- Dependencies should be permissive (MIT/BSD/Apache) where practical — prefer
  what is already in the stack over adding something new.
- Commit messages in this repo carry no `Co-Authored-By` trailer.
- Don't commit or push unless asked.
