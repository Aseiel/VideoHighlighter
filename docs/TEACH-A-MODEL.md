# Teach it something new: from a few videos to your own model

The manual version of this worked: download a few eight-minute videos, cut
them into five-second samples with the action cropper, sort the samples into
folders, label them, and a model trained on about a hundred samples found the
thing well. `modules/teach` is that process with the tedious parts automated
and the judgement parts made quick, driven from one command line, so a person
or an LLM agent can run it.

```
say what to find ─► add videos ─► cut ─► (focus) ─► sort ─► review ─► build ─► train
     add-class        add-video                      ▲         │
                                                     └─────────┘  accepted samples sharpen
                                                                  the next sort
```

Two kinds of model:

| `--task` | a sample is | trains | the app then uses it as |
|---|---|---|---|
| `actions` | a 5 s clip, labelled as a whole | R3D (`model_training/r3d`) | the custom R3D action model |
| `objects` | a place to look; labels are boxes | YOLOX (`training/train_yolox_run`) | a custom detector in `models/custom/` |

Everything lives in one project folder (by default `<user data>/teach/<name>`),
never in the repository. Every step is safe to re-run, and `status` always says
which one comes next.

## The quickest way

First, on a new machine:

```bash
python -m modules.teach --project my-first doctor
```

It checks in seconds that OpenCV, ffmpeg, CLIP, disk space and a training
device are there, and says how to fix anything that's missing. `quick` runs
the same checks and stops before creating anything if something required is
missing.

Make a folder with one subfolder per thing to find, named after it, holding a
few short clips that show it (3-5 is plenty). Then:

```bash
python -m modules.teach --project my-first quick --task actions \
    --examples ~/examples/ --videos ~/Videos/footage/
```

```
examples/
  <first thing>/    a.mp4  b.mp4  c.mp4
  <second thing>/   a.mp4  b.mp4
```

That creates the project, makes the folder names the classes and the clips
their first examples, adds the footage, and runs everything that needs nobody:
cut, crop (with `--focus`), sort, auto-accept, boxes, build. It stops at the
first point where someone has to look, which is nearly always "check these
guesses". Then:

```bash
python -m modules.teach --project my-first review --window
```

Each tile already shows its guess. Click the wrong ones, then press Enter. It
re-sorts with what you just confirmed and shows the next batch. When `status`
says it's ready: `auto --train`.

### How little checking it takes

After a class has **5 checked samples**, confident guesses for it are accepted
automatically. They must look as much like the checked ones as those do like
each other, and clearly unlike every other class. The same happens for "none
of these" once five have been marked. What's left for you is the doubtful
ones, plus a few **spot checks** of what was auto-accepted on every batch. If
spot checks overturn more than 20% for a class, auto-accept switches off for
it and hands everything it decided there back to you. Before training, each
class needs a few spot checks (10% of its auto-accepted samples, at least 3).
The held-out samples that score every round are always ones a person
checked, so the score can't be flattered by the auto-labels.

Settings: `set auto_accept=false` to check everything by hand;
`auto_min_checked`, `auto_gate`, `auto_margin` and `auto_max_error` tune it.

## A whole run, step by step

```bash
T="python -m modules.teach --project my-first"

$T init --task actions            # add --focus to crop samples to the people in them
$T add-class "<name>" --description "<what it looks like, in a few words>"
$T add-video ~/Videos/clip1.mp4 ~/Videos/folder-of-clips/ https://example.com/video
$T cut                            # 5 s samples, frame-accurate
$T add-example --class "<name>" --clip ~/examples/good-one.mp4   # optional, strongly helps
$T sort                           # every sample scored against every class
$T review                         # draws review/sheet-0001.jpg
$T verdict --sheet 1 --accept 1-9,11 --reject 10 --negative 12 --relabel 13="<other class>"
$T sort                           # re-sort: what you accepted is now an example
$T review                         # ... repeat until status says to build
$T build
$T train                          # installs the model only if it beats the last round
$T status
```

Each command prints one JSON object. `status` returns `next.command`, the exact
command to run next, and `next.who`: `auto` for steps that run unattended,
`judge` for steps where someone has to look.

## Naming a class

A name becomes a folder, a label in the timeline, and the words CLIP sorts
with before any model exists, so:

- lowercase words separated by spaces, like the stock labels. Name actions by
  what is being done; name objects with a singular noun;
- words that describe it, not a code: CLIP cannot sort by `cls1`;
- a `--description` with more detail. It is used for sorting, not shown.

`check-name "<name>"` runs the rules (duplicates, near-duplicates such as
plural forms, reserved or unsafe characters, vague words). To let examples
suggest the name:

```bash
$T suggest-names --clip a.mp4 --clip b.mp4 --clip c.mp4
```

It ranks the labels the stock models already know (Kinetics-400 for actions,
COCO for objects) by how well they describe the clips, and returns two checks:

- `fit: "good"` → reusing that name is sensible: it means the same thing
  everywhere, and the stock model can help later;
- nothing "good" is the usual case for something new. Pick your own name in
  the same style;
- a low `consistency`, or a `split` with two groups of clip indices, means the
  examples look like two different things. Two classes will train better than
  one class that means two things.

Three to five examples are enough for this, and they are also the best start
for sorting.

## Reviewing

`review` puts up to 24 samples on one image: each tile is numbered, shows the
start, middle and end of the clip (one frame for objects), and is captioned
with the guess. The picks are, in order: samples where the last model and CLIP
disagree, samples nearest the decision line, a few candidate negatives, and
always a slice of confident guesses. Answer with:

| flag | meaning |
|---|---|
| `--accept 1-5,7` | the guess is right |
| `--relabel 8="<class>"` | the guess is wrong, and this is what it is |
| `--negative 9` | shows none of the classes (teaches the model what to ignore) |
| `--reject 6` | unusable: unclear, a bad cut, something else entirely |
| `--accept-rest` | every tile not mentioned is taken as guessed |

Prefer folders? `sort --folders` lays out `sorted/<class>/`, `sorted/_unsure/`,
`sorted/_none/` and `sorted/_reject/` as hard links. Drag the wrong files to
the right folder, then `folders --read --confirm all`. A moved file counts as a
decision wherever it lands. An unmoved file counts only in the folders you
`--confirm`.

Re-run `sort` after each sheet or two: accepted samples become that class's
examples, so the next guesses are better and the reviews get faster.

## Object projects: boxes

After samples are accepted, boxes are needed:

```bash
$T boxes propose        # stock detector + CLIP pick a box on each accepted sample
$T boxes review         # boxes-0001.jpg: is the yellow box around it, and tight?
$T boxes verdict --sheet 1 --accept 1-12 --reject 13
$T boxes worklist       # samples with no good box: label these in tools/labeler.py
$T boxes import exports/*.json --accept-all
```

For a class the stock detector already knows, its own box is used, and at
confidence 0.6 or more it is accepted without review. Otherwise each detected
region is compared with the class, and the one that stands out wins. For
something no detector knows, CLIP scans overlapping regions of the frame, then
shrinks the best one while it still looks like the class. Those boxes are
coarser and always go to review. Once a class has three accepted boxes, their
crops replace its name as what "looks like it" means, which makes every later
proposal better. What still can't be found goes to the labeller, as before.

## Honest numbers

About 100 accepted samples per class gave good results by hand. The default
target is 100, and 20 is the minimum `status` will train on. A first model is
reliable on footage like its samples and weaker elsewhere; the way to improve
it is **new footage**: `add-video` something it has not seen, then cut, sort
and review. Its mistakes there are the most useful labels there are. Every
round is scored on the same held-out samples (chosen once, never trained on),
and a round is installed only if it beats the installed one. Earlier rounds
stay in `runs/`.

## For agents

Loop on `status` and run `next.command`. A `judge` step needs vision or a
person: open the sheet image named in the `review` output, look at every tile,
and answer with `verdict`. Never accept a tile you have not looked at, and
never use `--accept-rest` without looking. Every safeguard after this step
assumes someone looked. `.claude/skills/teach-a-model/SKILL.md` has the
agent-facing version of this page.

Content-neutral by design (CLAUDE.md): classes are the user's own data. Do not
add preset class lists or example categories to the repository.
