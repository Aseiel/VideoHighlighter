# Boxes or outlines

A detector answers with boxes. The player overlay can show those boxes, or
the **outline** of what is inside each one: switch **▭ Boxes / ✏️ Outlines**
next to the overlay's Filter button.

## What it is for

**Understanding how accurate a trained model is.** A box says roughly where
the model found something. An outline shows what it actually picked out: one
of the two people a box spans, the object and not the hand holding it, or a
patch of background it mistook for the thing. Wrong detections that hide
inside a reasonable-looking box are easy to see as a shape. That is the point
of the switch: checking a model you trained before you rely on it or share it.

An outline is not something the detector drew. It is a second model's reading
of the detector's box: Segment Anything (SlimSAM, Apache-2.0), prompted with
the box. So the box stays on screen, dashed, behind every outline, and where
the two disagree, the box is what the detector said.

## How it works

- Outlines are traced in the background for the detections around the
  playhead, nearest first, on the exact frame the detector saw (read from the
  file, not grabbed from the screen). Seeking drops work the playhead has left
  behind.
- They are written into the analysis cache next to the boxes (`contours`), so
  a composition rule with `outline: true` reuses them instead of tracing again
  (see [the outlines plan](plans/2026-09-26-composition-outlines.md)).
- The first time in a session, loading the model takes tens of seconds; the
  label next to the switch says so. After that, one frame takes about 0.25 s
  on an Intel Arc GPU, and about 1.4 s on a CPU.
- If SAM cannot be loaded (no weights yet and no network), outlines fall back
  to GrabCut, which needs no model but often finds nothing.
- The choice is remembered.

## Benchmark: do outlines make rules more accurate?

Measured 2026-10-16 on one user-taught detector with three part classes
(a source `s` and two region classes `a` and `b` that sit side by side in the
frame), over 666 clips in 14 user-defined categories, 3,042 frames. Each
clip's category was the ground truth. Decisions were made by the composition
engine itself on each geometry.

**Does the outline sit on the part?** A labelled point inside the part should
be inside its outline. (Rows are the three classes.)

| | box | SlimSAM outline | SAM ViT-base outline | GrabCut outline |
|---|---|---|---|---|
| class 1 | 93% | 79% | 79% | 69% |
| class 2 (small) | 91% | 63% | 64% | 62% |
| class 3 | 92% | 81% | 74% | 70% |
| detections outlined | — | 100% | 100% | 31% |

**Do decisions get better?** Threshold-free separation (AUC, 0.5 = chance) of
the clips where a rule should fire from those where it should not:

| task | boxes | SlimSAM | SAM ViT-base |
|---|---|---|---|
| `s` on `a`: clips of `a` vs clips of `b` | 0.757 | 0.777 | 0.768 |
| `s` on `b`: clips of `b` vs clips of `a` | 0.574 | 0.576 | 0.592 |
| `s` on `a` vs clips with no contact | 0.883 | 0.894 | 0.891 |
| `s` on `b` vs clips with no contact | 0.846 | 0.823 | 0.817 |

With thresholds, the best outline relation matched today's box test but did
not beat it overall. Where boxes and outlines disagreed on a frame, boxes
agreed with the ground truth more often in 5 of 7 rule/relation pairs.

**Reading it:**

- On compact parts that sit next to each other, outlines do **not** make rule
  decisions more accurate. About a third of the outlines on the smallest class
  landed on the larger surface around it, and a SAM ten times bigger did not
  fix that.
- What limited accuracy was the detector: in one category it found both parts
  in only 17% of frames. No geometry can fire on a part never found. Better
  training data is the bigger lever.
- What did help was `compete_with` (a source counts only for the region class
  it is closest to): `s` on `b` went from 0.57 to about 0.77, on plain boxes.
- Outlines still pay where boxes are badly wrong: diagonal, reaching or
  L-shaped things, whose boxes are mostly empty (box IoU with the real shape
  0.20–0.47 on synthetic shapes).

So: use rules on boxes (with `compete_with` where regions touch) for
decisions, and outlines for **seeing** what a model found.

| | per frame |
|---|---|
| detector (YOLO-size, CPU) | 33 ms |
| SlimSAM, Intel Arc | 219 ms |
| SAM ViT-base, Intel Arc | 282 ms |
| SlimSAM, CPU | 1,356 ms |
| GrabCut, CPU | 82 ms |

Requested by Neva; the idea of outlines in place of boxes came from Szymek.
