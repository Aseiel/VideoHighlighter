---
name: teach-a-model
description: Train a custom VideoHighlighter action or object model from a few videos the user provides — cut into samples, sort with CLIP, review contact sheets, build, train, install only if better. Use when the user asks to teach, label, or train the app to recognise something new.
---

# Teach a model

The pipeline is `python -m modules.teach` (docs/TEACH-A-MODEL.md). Every
command prints one JSON object; `status` names the next command.

## Loop

1. `python -m modules.teach --project <name> status`
2. Run `next.command`, filling any `<placeholder>` from what the user said.
3. Repeat. Stop and ask the user when:
   - no classes exist and the user has not said what to find;
   - `next.who` is `judge` and you cannot see images;
   - a command returns `error`, or training fails (read `runs/<n>/train.log`).

## Fastest start

If the user has example clips, ask them to put them in one subfolder per
class, named after what it shows, then run
`quick --task <actions|objects> --examples <folder> --videos <files/folders>`.
It runs every unattended step. After a judge step, `auto` continues
(`auto --train` includes training). A person reviews fastest with
`review --window`. Tiles show their guesses; they click the wrong ones and
press Enter.

## Starting a project

- Ask what to find and for videos (files, folders or URLs) if not given.
- `init --task actions` for something that happens over time (a movement, an
  activity); `init --task objects` for a thing visible in one frame.
- Name classes with `check-name` first. With example clips, run
  `suggest-names --clip ...`: reuse a `fit: "good"` stock label; otherwise
  choose a descriptive name in the same style, and always add
  `--description`. If `split` is not null, tell the user the examples look
  like two things and propose two classes.
- Put the user's example clips in with `add-example`: they make sorting far
  better than words alone.

## Reviewing (judge steps)

- `review` returns `image`: open it and look at every numbered tile. Tiles
  show start / middle / end of a clip; the caption is the guess.
- Answer with `verdict --sheet N` using `--accept`, `--relabel N=<class>`,
  `--negative` (none of the classes) and `--reject` (unusable). Decide every
  tile you can see. Use `--accept-rest` only after checking each unmentioned
  tile really matches its caption.
- When unsure about a tile, reject it: a wrong label hurts more than a
  missing one.
- Run `sort` again after every one or two sheets (the window does it for you).
- Tiles captioned "spot check" were auto-accepted. Judge them as strictly as
  any other tile: they are how auto-accept is kept honest, and training waits
  until each class has a few.

## Rules

- Never edit `project.json` / `samples.json` by hand; use the commands.
- Never train with `--install always` unless the user asks: the default
  installs only a model that beats the installed one on the held-out set.
- Report results as the per-class accuracy from the round's metrics
  ("finds <class> in 7 of 10 held-out clips"), not as a single score.
- Content-neutral: never add class names or presets to the repository.
