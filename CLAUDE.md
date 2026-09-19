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

## Models people make and share

The app's direction is a community of small models: people teach it what to
find in their own footage, train a model, and share it. Two rules make that
possible, and both are load-bearing.

**No AGPL detector, anywhere the app runs.** A model trained on top of one
inherits its licence, and a shared model has to be usable by anyone, in any
build. Object detection is YOLOX (Apache-2.0) through OpenVINO;
`tests/test_no_agpl_detector.py` fails if `ultralytics` or its model files come
back into app code. Dev-only `tools/` may use it behind a guarded import.

**A shared model carries the model, never the material.** Training on footage
and handing that footage to others are different acts, and the second is where
the copyright risk is. `model_hub/` enforces it: a package is exactly
`model.onnx` + `videohighlighter.json` + a generated `README.md` (+ `LICENSE`),
detectors must be YOLOX-layout, licences are permissive only, and the publishing
checklist must be confirmed before `hub.release_the_kraken` clears an upload.
Don't add a field or file that could carry frames, crops, clips, audio or paths.
`model_hub/` is kept identical to the Pro edition's copy.

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
  already pushed to `main` — save for the one narrow exception below.
- A contributor's authorship is theirs. Don't "fix" it by re-authoring their
  commits or by adding `Co-Authored-By` to a merge commit — the author field
  already credits them, and a trailer is strictly weaker.
- If someone is missing from the contributors sidebar, that is almost always the
  stale cache rather than the repo. Check the commit's `author.login` via the
  API first. The answer is to wait for the recompute or to open a support
  ticket — never to push again harder.

### The one exception: our own tip commit, minutes old

A message is part of a commit's hash, so fixing a message means re-hashing. When
the commit is the *tip* of `main` and every condition below holds, that re-hash
touches nothing the section above is protecting: no commit is built on it, and
no contributor's hash moves.

- It is `origin/main`'s tip — no descendants, on no other branch. Check with
  `git log --oneline <sha>..origin/main` (empty) and `git branch -a --contains
  <sha>` (only `main`).
- Every commit being rewritten is authored by a maintainer. One outside
  contributor anywhere in the range and the exception is void, whatever a
  trailer claims — read the `author` field, not the message.
- It was pushed minutes ago and nobody has pulled or branched from it.
- Only the message changes. `git diff <old> <new>` must come back empty.

Then `git commit --amend` and `git push --force-with-lease origin main`. Never a
bare `--force`: the lease is the part that proves nobody pushed in between.
Re-check the conditions immediately before pushing rather than when you started,
because `origin/main` moves under you.

Miss one condition and the rule above stands: leave it. Even when it works this
costs something small and permanent — a PR merged as the old hash keeps pointing
at a commit that is no longer on `main`, and that never heals.

## Conventions

- The packaged exe is `--windowed`: `stdout` goes nowhere, so
  `modules/system/debug_console.py` tees all output to `debug.log` and the
  optional "Debug log" window. Diagnostic output belongs in `print()`
  (→ debug log); `append_log()` is the user-facing log pane and is only for
  things the user acts on.
- Dependencies should be permissive (MIT/BSD/Apache) — prefer what is already
  in the stack over adding something new. Check the licence of a model's
  *runtime and training toolkit*, not just its weights.
- Commit messages in this repo carry no `Co-Authored-By` trailer. Amending it
  off a pushed branch does not help: GitHub's squash-merge box prefills
  co-authors from *every* commit a PR has ever had, force-pushed-away ones
  included, so it reappears in the merge commit. Clear it in the merge box,
  or keep it out of the first commit.
- Don't commit or push unless asked.
