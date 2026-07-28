# Choosing a detector

VideoHighlighter ships four detection engines. They are not ranked — each
answers a different *kind* of question, and picking the wrong one wastes hours
before you find out. This guide is about matching the question to the engine.

Everything here describes mechanisms. What you point them at is your business:
queries and rules are supplied at runtime and stored as your own data, never in
the application.

---

## The one-minute version

| Your question | Engine | Training needed? |
|---|---|---|
| "Where is this thing in the frame?" — a bounded object you can point at | **Object recognition** | Yes, unless it's one of the 80 COCO classes |
| "What is happening here?" — motion, an activity unfolding over time | **Action recognition** | Yes, for anything outside Kinetics-400 |
| "Find where the video looks like X" — scene, setting, framing, mood | **CLIP search** | No |
| "Find where several of the above hold at once" | **Composition engine** | No — it combines what the others produce |

If you are new: start with CLIP search to explore what's in the video, then
reach for a trained detector only for the handful of things you need to be
fast and reliable about.

---

## 1. Object recognition (YOLO)

**Answers:** where is this thing, right now, in this frame.

**Speed:** real time.

**Vocabulary:** the 80 COCO classes out of the box, plus any model you train
and import. Runs a `.pt` model directly or an OpenVINO export (`yolo11n_
openvino_model/` by default), whichever you point it at.

**Strengths.** Precise boxes, stable confidence, cheap enough to run over a
whole video. Because it emits boxes it can *count*, and counting is exact:
"how many" is the number of boxes, "none" is zero boxes.

**Limits.** It only knows what it was trained on. If your subject isn't in
COCO, you are training a model — see "Training your own class" below.

**Use it when** the thing matters enough to justify labelling, or you need it
at frame rate.

---

## 2. Action recognition

**Answers:** what kind of motion is happening across this stretch of time.

**Speed:** windowed — it classifies a clip of frames, not a single frame.

**Vocabulary:** Kinetics-400 (400 everyday actions), plus custom models you
train from folders of example clips. Backbones are torchvision video networks
(`r3d_18`, `mc3_18`, `r2plus1d_18`).

**Strengths.** It is the only engine that sees *time*. Motion, rhythm, and the
shape of an event over several seconds are invisible to everything else here.

**Limits — read this one before you build a dataset.** The model is fed an
ROI crop, not the whole frame: a detector finds the region of interest and the
network sees that zoomed region. Global spatial layout is cropped away before
the model gets a vote.

The practical consequence: **it cannot separate classes that differ only by
*where* something happens.** If two of your categories involve the same motion
in different places, the model has no access to the information that
distinguishes them, and training them as separate classes will just produce a
confused model and a smaller dataset per class.

The fix is to merge them into one class and let the composition engine split
them by location afterwards. See "Primitives, not categories".

**Use it when** the thing you want is defined by movement rather than
appearance.

---

## 3. CLIP search

CLIP embeds a whole image into a vector, then compares it against a phrase you
type. Everything it is good at follows from that, and so does everything it is
bad at.

**The cost model is what makes CLIP special.** It embeds each frame *once*,
into an index. After that, unlimited queries are nearly free. Every other
engine here charges you per query, per frame.

**Use it for** setting, scene type, framing, time of day, general mood — things
that describe the picture as a whole.

**Limits.** CLIP scores *scenes*, not objects. If your target is a few percent
of the frame, the embedding is dominated by everything around it, and you are
effectively asking "does this look like the *scene* where that appears", not
"is it here".

So: **large or scene-defining → works well. Small discrete object → use the
detector instead.**

**On the score.** Raw CLIP similarity has no meaningful zero — everything lands
in a narrow band. The prefilter compares your query against generic negative
prompts so the result is a calibrated softmax rather than a bare cosine, which
is what makes a threshold transferable between videos. Judge results by
*ranking* first; treat the absolute number as secondary.

---

## 4. The composition engine

The other three engines produce detections. This one turns detections into
*meaning*, using rules you write in `composition_rules.yaml` (it lives in your
user data folder, never in the app).

A rule counts how many boxes of one class have their centre inside a box of
another class, and fires when the counts hold steady over a short window:

```yaml
events:
  - name: held_object
    label: Held object
    window_secs: 0.75      # majority-vote smoothing
    persist_secs: 0.5      # keep a box alive this long after it disappears
    rules:
      - {source: handle, region: tool, min_count: 1}
```

`min_count` / `max_count` give you counting *and* absence — `max_count: 0`
means "none of these inside that". Source boxes are consumed across rules, so
two rules each needing one source genuinely require two distinct objects.

### Two settings worth understanding

**`persist_secs`** keeps a class alive after its last detection. Raise it when
a region gets occluded at the exact moment you need it. But raise it only on
rules that need it: a *moving* object whose box drifts far enough will fail the
overlap match and be tracked as a second instance, inflating your counts and
corrupting `min_count` rules.

**`window_secs`** is smoothing, not timing. Note what this means: every rule is
a *state* test, evaluated per frame. It answers "is this true now", not "did
this just start". For something that appears and then stays on screen, a rule
will keep firing for as long as it remains visible.

---

## Primitives, not categories

The single most useful principle in this document.

When you have several categories that share the same appearance and differ only
by *where* something is, **do not train them as separate classes.** You will be
asking the network to learn a distinction its input doesn't contain, you will
split your training data N ways, and every new category will mean relabelling.

Instead:

1. Train **one** class for the thing itself — the primitive.
2. Get the location from something you already have (another trained class, a
   detector you already run).
3. Write one composition rule per category.

Adding a category then costs a few lines of YAML instead of a training run.
This also means a small number of well-trained primitives goes a very long way:
three good classes can express a dozen composed events.

---

## Training your own class

When no engine above can see your subject, you train a detector. Briefly:

- **Box the region, not the speck.** Small objects are the hardest case. A
  larger, stable region containing the thing beats a tight box around a few
  pixels, tracks better between frames, and works with the composition engine's
  centre-inside test.
- **Hard negatives matter more than more positives.** Include frames
  containing whatever your detector will confuse for the target, with *no* box
  on them. Budget roughly a third of your set for this. Verify your conversion
  step keeps zero-annotation frames — many drop them, and those are the ones
  doing the work.
- **Diversity beats volume.** 2000 instances from 300 sources beats 10000 from
  20. Adjacent video frames are near-duplicates; sample at least a second apart.
- **Write your labelling rule down first.** Inconsistency — the same appearance
  boxed in half your frames and not the other half — caps accuracy harder than
  dataset size.
- **Label the hard state too.** If a region changes appearance during the event
  you care about, and you only ever labelled it clean, your detector will drop
  out at exactly the wrong moment.
- **If two architectures score the same, you are data-limited, not
  model-limited.** Stop swapping backbones; fix the data.

**On licensing:** this edition is AGPL-3.0 and uses ultralytics YOLO, which is
also AGPL-3.0, so you can train and ship models with the ultralytics tooling
freely — as long as whatever you distribute is AGPL too. That is the normal
case here and needs no special handling.

---

## Cost at a glance

| Engine | Per-frame cost | Repeat queries |
|---|---|---|
| Object recognition | real time | re-run required |
| Action recognition | windowed, fast | re-run required |
| CLIP index | fast, **once** | nearly free, unlimited |
| Composition engine | negligible | instant — it reads cached detections |

The practical pattern for anything expensive: build the CLIP index first, use
it to narrow down where to look, then run the costly engine only on those
stretches.

---

## Command-line helpers

Both CLIP entry points have a standalone CLI, useful for testing a query before
committing to a full run.

CLIP index — `--query` is repeatable, and every query after the first is nearly
free because they all score against the same index. This is the cost model from
the table above, made visible:

```bash
python -m llm.clip_index --video "v.mp4" --interval 2 --query "an outdoor scene" --query "a close-up" --topk 10
```

Pass `--cache` to reuse an index you already built instead of re-embedding.

Single-query ranking without keeping an index:

```bash
python -m llm.clip_prefilter --video "v.mp4" --query "an outdoor scene" --topk 20
```

Add `--help` to either for the full option list.

---

## Quick troubleshooting

| Symptom | Likely cause |
|---|---|
| CLIP ranks obviously wrong frames highly | Target too small in frame — CLIP sees the scene, not the object. Use the detector. |
| CLIP scores look uniformly low | Normal. Judge by ranking, not by the absolute number. |
| A composition rule never fires | One of its classes isn't being detected. Check each class fires on its own before combining. |
| A rule fires far longer than the event | Rules test state, not onset — it stays true while the thing is visible. |
| Trained model confuses two categories | They probably differ only by location. Merge and split with a rule. |
| Action recognition disabled at startup | PyTorch or torchvision isn't installed; the R3D backend needs both. |
