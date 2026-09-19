# Training a model of your own

Design note for the training tab. Nothing here is built yet except where it
says otherwise — several stages already exist as scripts and need wiring, and
one stage does not exist at all.

## The flow, as a user describes it

> I have a movie. I care about three things in it. I don't have a model. So I
> mark the three things, ask the app to train on them, it finds them through
> the whole movie, trains, shows me progress — and I end up with my own small
> model.

That is the right shape and it is achievable. Two steps in it are doing more
work than the sentence suggests, and the design lives or dies on how honestly
they are handled:

- **"it finds them in the whole movie"** — it finds *candidates*. Some are
  wrong. Something has to separate them, and that something is a person,
  briefly.
- **"and then we have our own mini model"** — a model that works on *this*
  footage. How far it travels beyond it is the single biggest expectation to
  set, and it is set at the start or it is a support ticket at the end.

## The pipeline

### 0. Seed — the user marks each thing once

Draw a tight box, give it a name. Already exists twice over:
`video_ai_editor/live_category.py`'s `teach()` (box in frame pixels → crop →
embed → fold into the category) and `llm/example_search_ui.py`'s
`RegionSearchDialog`.

**These seed boxes are the best labels in the whole dataset** — a human drew
them, tight, on purpose. They go into the training set directly. Nothing later
in this pipeline produces geometry this good, so none of it should overwrite
them.

Three or four seeds per class beats one. Averaging examples that are genuinely
alike tightens the prototype; the cost is one more drag.

### 1. Mine — score the whole file, keep candidates

Sample the video at an interval, embed a grid of overlapping sub-regions per
frame, score each against every taught category, keep the frames where one
clears the gate. Output per candidate: timestamp, class, coarse box, score.

**Use the live path's scoring, not the offline path's.**
`video_ai_editor/live_category.py:166` scores by relative standout across tiles
of the same frame, gated against an absolute floor, with constants fitted to
measured data. `llm/example_search.py:224` still uses softmax against a
background embedding — which the live path measured as collapsing to ~0% and
deliberately abandoned. Porting that scorer into the offline scan is a
prerequisite for this whole feature, not a nice-to-have.

Two settings that differ from live use:

- **Mine at a lower gate than the overlay uses.** Live scoring is tuned for
  precision, because a "seen" range should be trustworthy. Mining wants recall
  — the review step is the precision stage, and a candidate a person rejects
  costs one click, while one never proposed costs a class the model never
  learns.
- **Deduplicate by embedding.** Adjacent samples of the same shot are
  near-identical and teach a detector nothing, while inflating the review queue
  they have to click through. The embeddings are already computed: drop a
  candidate whose cosine to an already-kept candidate is above a cut. This is
  the difference between reviewing 200 frames and reviewing 2,000 that are the
  same 200.

### 2. Refine — coarse region to usable box

**This stage does not exist and it is the real gap.**

The mined box is whichever fixed sub-region won: always the same size, at one
of nine positions. It says which part of the picture to look at. It is not
fitted to anything. Train on it directly and the model learns that the class is
a half-frame blob at nine possible locations.

Two candidate refiners, both needing measurement before either is trusted:

- **Prompt a detector.** `llm/owl_detect.py` (OWLv2, Apache-2.0 weights and
  runtime) takes a plain noun and returns a fitted box per instance. Best
  quality, works only when the class has a name that means something to a
  general detector, and costs on the order of seconds per frame — acceptable
  here, because it runs on accepted candidates only, not on every sampled
  frame.
- **Shrink-search.** Re-embed progressively smaller crops inside the winning
  region; keep the smallest crop that holds its cosine to the category vector.
  No new dependency, no vocabulary requirement, uses only what is loaded. Cost
  is a handful of extra embeds per candidate. Unvalidated — this is the first
  thing to measure.

Fall back to the seed box's size and aspect anchored on the winning region when
neither refiner produces anything. A consistent approximation is more trainable
than a mix of good and wild boxes.

### 3. Review — the step the flow skips

A grid of proposed crops, biggest-doubt-first, with three actions: accept,
reject, drag to fix. Keyboard-driven, because the whole value is that it takes
minutes.

Design rules that decide whether people actually do it:

- **Show crops, not frames.** The question is "is this the thing", and a crop
  answers it in a glance where a full frame does not.
- **Review the confident ones too.** Sampling only the uncertain leaves
  confident-and-wrong labels in the set, and those poison training hardest.
  Mix in a fraction of the high scorers — a tenth is a reasonable start.
- **Show the count against the target.** "148 of ~200 for this class" tells
  someone whether they are nearly done. An unbounded queue reads as infinite
  and gets abandoned.
- **Let it be resumed.** The verdicts are a file. Closing the app mid-queue
  must not lose them.

Honest cost to state up front in the UI: a few minutes per class. Not zero.
Zero is not on the menu, and promising it is how the feature gets a reputation.

### 4. Assemble — dataset on disk

Accepted candidates plus seed boxes, converted to the layout
`training/train_yolox_dataset.py` already consumes (YOLO-format labels → COCO
via `tools/convert_yolo_to_coco.py`), split train/val.

**Include negatives.** A detector trained only on frames containing the classes
never learns what they are not, and fires everywhere. Sample frames where every
category scored low — mining already knows which those are, for free. Start at
roughly one background frame per three positives and measure.

Dataset lives in user data, never in the repo.

### 5. Train — with progress

`training/train_yolox.py` today prepares the dataset, writes a YOLOX experiment
file and the `labels.json` sidecar, and then **prints the commands** for the
user to run inside a separately cloned YOLOX repo. That is the gap between a
developer script and a feature.

What has to change:

- **Training runs from the app.** Either vendor the YOLOX training code
  (Apache-2.0, so this is permitted) or drive it as a subprocess we own.
- **Progress is reported.** Epoch, loss, val mAP, elapsed and estimated
  remaining. Cancellable. If it is a subprocess, its output is parsed; if
  in-process, a callback per epoch.
- **Refuse early on CPU.** Training a detector on CPU is not viable and the
  user must be told before the run, not twenty minutes into it. Check the
  device up front and say so plainly.
- **Start small.** YOLOX-tiny or -nano. The goal is a fast model that knows a
  few classes, not a general detector.

### 6. Export and install

ONNX without decode baked in → OpenVINO IR → `models/custom/`, with the
`labels.json` sidecar written beside it. All of this exists in
`training/train_yolox.py`; the sidecar in particular is load-bearing — without
it the app finds zero class names, logs "No class names for custom model", and
silently falls back to the 80-class detector.

### 7. Use, and measure whether it worked

The new model becomes selectable as the detector. Then re-run the same video
and compare detection counts against the run before it — the report already
carries them. A model nobody measured is a model nobody can improve.

## The loop

Everything above describes one pass, and one pass is not the feature. The
first model is a starting point that costs a few minutes of clicking; what
makes it worth having is that each round after it is cheaper than the one
before, and better.

### Round 2 onward: the model becomes the proposer

After the first training run there is something better than CLIP tiles for
finding candidates — the model itself. It runs at detector speed, and it
returns **fitted boxes**, which means the refinement stage that has no
known-good answer largely stops mattering from here. Round 1 exists mostly to
buy a proposer good enough to bootstrap round 2.

So the loop is:

```
seeds ──► mine ──► refine ──► review ──► train ──► model
                                 ▲                    │
                                 └──── propose ◄──────┘
```

CLIP does not leave the loop. It keeps running alongside the model, because
the two are wrong about different things, and where they disagree is where the
information is.

### What to put in front of the human next

The review queue stops being "everything the miner found" and becomes a
selection. Ranked by expected value, most useful first:

1. **Disagreements.** The model detects where CLIP scored low, or CLIP scores
   high where the model found nothing. One of the two is wrong and the answer
   is worth more than any confident agreement.
2. **The model's low-confidence detections.** The classic uncertainty sample —
   the decision boundary is where a label moves the model most.
3. **Suspected misses.** Frames CLIP likes and the model returned nothing for.
   These are the labels that fix recall, and a model never proposes them by
   definition, so nothing else in the loop will surface them.
4. **A sample of confident agreements.** Small, but never zero. It is the only
   check on both engines being confidently wrong together, and it is what
   detects drift before it compounds.

### The rules that keep it honest

These are the parts that are easy to skip and expensive to skip.

- **Hold out a set that is never trained on.** Fixed at round 1, reviewed once,
  frozen. Every round is scored against it. Without this there is no answer to
  "is it actually getting better" — folding every reviewed frame into training
  feels productive and destroys the only measurement.
- **Never promote a worse model.** Train, score on the held-out set, and only
  install the new model if it beats the installed one. Keep the previous one.
  A loop that can go backwards without anyone noticing is worse than no loop.
- **Do not train on the model's own confident output.** Pseudo-labelling
  without review compounds the model's existing errors and the drift is
  invisible from inside the loop — every round agrees with itself more. If it
  is tried at all, it is at a high threshold, with a reviewed sample, and with
  the held-out score as the veto.
- **Record every round.** Which frames, which verdicts, which model, which
  score. A loop whose history is not written down cannot be debugged when
  round 6 is worse than round 4.

### New footage is how the loop escapes one file

The limit stated below — a model trained on one file knows that file's world —
is lifted by the loop and by nothing else. Add a second video, let the current
model propose, correct what it gets wrong there, retrain. Its errors on new
footage are exactly the labels that generalise it, and they are cheap to
collect because the model does the proposing.

This is the answer to "I want them detected in *every* video", and it should be
said that way in the UI: not one training run, but a model that gets better
each time it is pointed at something new.

### How much of this runs unattended

Mining, refinement, training, scoring and promotion all can. Review cannot, and
should not — every safeguard above rests on a person having looked.

So "auto" means the app does everything up to a review batch, then stops and
says a batch is ready. The user's total involvement per round is a few minutes
of clicking; everything either side of it is unattended. Worth building for
directly: rounds should be resumable, cancellable, and safe to leave running.

## What one round actually produces — measured

A YOLOX-tiny trained on roughly **100 labelled samples**:

- **Hit rate is wide: 30–100%, typically 50–100%.** The spread is the finding,
  not noise around an average. At this dataset size the model is reliable on
  the conditions its examples covered and unreliable outside them, so the
  number a user sees depends almost entirely on which frames went in — which
  is the same thing this document says about collecting from failing
  conditions, arriving from the other direction.
- **Localisation is the strong part.** Where it fires, the boxes mostly sit
  where they should. Detection rate and box quality are separate results here,
  and the weaker of the two is detection rate.

Two things follow for the design.

**A hundred samples is enough to be useful and not enough to be dependable.**
That makes it a good first round and a bad stopping point, and it is the
concrete argument for the loop: a user who trains once and stops lands
somewhere in that range with no idea where, while a user who does a second
round on new footage moves up it. The UI should treat round 1 as *begun*, not
*done*.

**Report the hit rate, never a single quality score.** A model that finds the
thing half the time is genuinely useful for jumping around a file and
genuinely useless for an unattended cut, and only the measured rate tells
those apart. Since the boxes are good where they fire, the honest sentence to
show is about how often it finds the thing — not about how well it frames it.

This also sets the expectation that ought to be in the UI before the first
run: *the first model will find some of them, not all of them, and it improves
each round.* Promising anything tighter is promising the top of a range whose
bottom is 30%.

*Sample is small and "hit rate" here is not yet pinned to a formal metric —
treat these as the working numbers until the held-out scoring of stage 7
produces comparable ones.*

## What this cannot do

Written here so it can be said in the UI, in the same words, before anyone
spends an evening on it.

- **A model trained on one file knows that file's world.** Same lighting, same
  camera, same rendering. It travels well across footage that shares those —
  more of the same source, the same series, the same capture setup — and badly
  to anything else. For "detect these in *this kind of* video, forever" that is
  usually enough. For "detect these anywhere" it is not, and the only route
  there is more rounds on more footage, not a better first run.
- **The model cannot be better than its labels**, and its labels come from an
  approximate refiner corrected by a person who was clicking quickly.
- **Rare classes stay rare.** If mining finds eleven examples, training on
  eleven examples produces a detector that has seen eleven examples. The count
  is shown for this reason.
- **It is not instant.** Mining is a whole-file embedding pass; training is
  GPU-minutes to GPU-hours. Both need real progress reporting, and the total
  should be estimated before the user commits.

## Where this lives in the UI

**Not a tab.** The app already has eight, and "too many options" is the loudest
complaint against it. A ninth called *Training* would also name the feature
after its implementation: nobody wants to train a model, they want the app to
find their things. The word "training" should not appear in the front of this
feature at all — it belongs in Advanced, next to the batch size.

The work happens at three different times, in three different postures, and
that is what decides the surfaces — not the fact that it is one feature
underneath.

### 1. Teaching happens on the video

Marking a thing means seeing it first, so the entry point is the player: drag a
box, name it. This already exists as a right-click in the live overlay.

Right-click alone is undiscoverable, so it needs one visible affordance
somewhere a person will find it — but the *action* stays on the video and does
not move into a settings screen.

### 2. A small panel is the feature's home

One list, beside the video, in the same dock the chat panel should live in.
Each row is one thing the user is teaching, and each row shows **state and one
action** — never a form.

The row is a state machine with exactly one next step:

| state | what the row offers |
|---|---|
| just named, no examples | show me one |
| has examples, never scanned | find these in this video |
| scanned, suggestions waiting | check 40 suggestions — about 3 minutes |
| enough checked | build it *(or start on its own)* |
| model exists | found in 9 of 10 places · add another video to improve |

One button per row, and it is always the thing to do next. That single property
is most of what makes this feel easy, because the user is never choosing — they
are answering. Everything the pipeline needs to decide (sample interval, gate,
model size, epochs, split ratio) is chosen for them and exposed only in
Advanced.

The panel does not exist until the user teaches their first thing. Before that
it is one empty-state line, not a screen.

### 3. Review is a mode, not a place

A review batch wants the whole window and the keyboard: a grid of crops,
accept/reject/fix, done. It is entered from the row's button, and it exits when
the batch is finished. It is not somewhere you can wander into and be confused
by, because it only exists while there is something to review.

### 4. Training is a job, not a screen

It runs in the background with the app's other long jobs. There is nothing to
look at and nothing to configure while it runs.

**Report progress in the user's terms, not the trainer's.** A bar and a time
estimate while it runs; afterwards a plain sentence derived from the held-out
score — how often it now finds the thing where a person said it was. Loss
curves, epochs and mAP are Advanced, or the debug log. A number nobody can act
on is not progress reporting.

### What the user is told they are doing

Not "train a model". Something closer to: *teach it what to look for, check its
guesses, and it gets better each time.* That description is accurate to the
mechanism, sets the expectation that checking is part of the deal, and never
promises a result the loop cannot deliver.

## Module layout

Keeping the repo's rule that a Pro-only feature goes in its own module, so the
shared files stay portable to the free edition:

| module | job | state |
|---|---|---|
| `llm/category_scoring.py` | tile scoring, shared by live and offline | **done** — ported out of `live_category.py` |
| `llm/example_mine.py` | scan a file → candidates | to do, no Qt, testable |
| `modules/box_refine.py` | coarse region → fitted box | to do, isolated so refiners can be swapped and measured |
| `modules/vision/label_store.py` | labels, verdicts, boxes on disk + COCO assembly | **done** — source-agnostic, segment-aware split |
| `modules/train_rounds.py` | round history, held-out set, promote/reject a model | to do, pure, testable without a GPU |
| `modules/review_queue.py` | what to ask the human next, ranked | to do, pure — the acquisition rules above |
| `training/train_yolox_run.py` | training with progress + cancel | **done** — fine-tunes on XPU/CUDA/CPU |
| `training/train_estimate.py` | how long a run will take, said before it starts | **done** — measured speeds per device, refined by every run on this computer |
| `modules/vision/training_preview.py` + `modules/ui/training_preview.py` | what the model finds each round on held-out frames; the "Watch it learn" window | **done** |
| `training/export_yolox.py` | checkpoint → ONNX → IR → `models/custom/` | **done** — raw-grid, layout-checked |
| `modules/ui/training_tab.py` | the Qt | to do |

### Where a box can come from

`label_store` records a `source` per box and branches on none of them, because
the sources differ enormously in quality and the good ones will change:

| source | geometry | limit |
|---|---|---|
| `hand` | a person dragged it | the best there is, and the slowest |
| `category` | winning region of a taught category | a fixed fraction of the frame at one of nine positions — says *where to look* |
| `labeler` | `tools/labeler.py` points, boxed at `box_frac` | same coarseness, for the same reason: it stores clicks, not extents |
| `prompt` | open-vocabulary detector | real geometry, needs a nameable noun |
| `model` | last round's detector | real geometry, no vocabulary limit — why round 2 is cheap |

So CLIP is the *finder*, never the box source. Its job is deciding which frames
are worth a look; something else always supplies the geometry.

`modules/box_refine.py` being its own module is deliberate: it is the stage
with no known-good answer, so it has to be replaceable without touching
anything around it.

## Constraints this design has to respect

- **No ultralytics, ever, in anything that ships.** YOLOX is Apache-2.0 and the
  existing training path avoids ultralytics on purpose. Moving training into
  the app makes the trainer a *runtime input*, so it leaves the `tools/`
  exemption behind: `tools/check_pro_boundary.py` must cover the new modules,
  and the frozen-bundle scan is what actually proves it.
- **No licence gate on this, for now.** Teaching the app a vocabulary of its
  own is the documented Pro boundary, so this feature sits on the paid side of
  it by default — but it ships ungated while it is being built and measured. A
  feature nobody can reach is a feature nobody reports bugs in. Revisit when
  the loop below closes and the quality claims hold up.
  *(This means "no licence check in the Pro build". Whether the whole stack
  also gets hand-ported into the free repo is a separate, larger call — it
  would move the boundary in `CLAUDE.md` — and is not assumed here.)*
- **Explanation is not.** Whatever the report says about what the model found,
  why a moment was chosen, and what to change next stays in both editions.
- **No subject matter in the repo.** Classes are named by the user at runtime
  and their names are their data — not in code, comments, tests, fixtures,
  presets, or commit messages.

## To measure before building far

1. Does shrink-search produce boxes tight enough to train on? If not, OWLv2 is
   the refiner and the feature inherits its "must have a nameable class" limit.
2. ~~How many accepted frames per class are needed for a usable YOLOX-tiny?~~
   Partly answered: ~100 gives 30–100% hit rate with good boxes (above). What
   is still open is where in that range a *second* round of the same size
   lands, since that decides whether the UI should suggest one more round or
   several.
3. What ratio of background frames stops the false positives without starving
   the positives?
4. How far does a model trained on one file actually travel? This is the claim
   the UI will make, so it should be a measurement and not a guess.
5. Does disagreement-ranked review actually beat reviewing a random sample of
   the same size? It is the standard result, but it is worth confirming here
   before the ranking is trusted — if it does not, the queue gets simpler.
6. How many rounds before the held-out score stops moving? That number is what
   the UI should tell someone to expect, and it decides whether "keep going" is
   ever the wrong advice.

## Build order

1. **Port the tile scorer to the offline scan.** Self-contained, and it fixes
   the confidence complaint on the existing feature whether or not any of the
   rest gets built.
2. **Mine + a label store**, output inspectable as JSON. No UI yet.
3. **Refine**, with both candidates behind one interface, and measure them.
4. **The review grid.** The piece that decides whether anyone finishes.
5. **Training with progress**, replacing the printed commands.
6. **Export, install, and the re-run comparison** that says whether it worked.

That is round 1. Then close the loop:

7. **Held-out set and round history** — frozen at round 1, so every later round
   has something honest to be scored against. Build this before round 2 exists,
   not after; retrofitting a held-out set means throwing away the rounds that
   contaminated it.
8. **The model as proposer**, running beside CLIP, with the disagreement
   ranking driving the review queue.
9. **Promote-on-improvement**, keeping the previous model.
10. **Point it at a second video** — the step that turns a one-file model into
    one worth keeping.

Stages 1 and 2 are useful on their own. Nothing before stage 5 needs a GPU.
Stage 7 is pure bookkeeping over files and needs neither a GPU nor a model, so
it can be built and tested in parallel with anything above it.
