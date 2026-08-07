# Detections and composed events

Two different things end up as tags on a clip, and telling them apart matters
when deciding what to change.

**Detections** come straight from a detector. One class, one box, one
confidence. The detector either has that class or it does not.

**Composed events** are produced by the composition engine from several
detections that co-occur — a rule saying "when these appear together, in this
arrangement, call it this". No detector has a class for a composed event; it
exists because a rule said so.

The report labels them separately, under `OBJECTS` and `EVENTS`.

## Why the distinction matters

The fix for a wrong tag is different in each case:

- **A wrong detection** is a detector problem — threshold, or a training gap.
  See `thresholds.md` and `training.md`.
- **A wrong event** is a rule problem. The underlying detections may all be
  correct while the rule combining them is too loose or too strict.

Chasing a rule problem by retraining a detector is a lot of work for no result,
and it is an easy mistake to make when both appear as plain tags.

## Rules that fire too often

A rule with a wide time window or a permissive count will fire on incidental
co-occurrences. Tighten the window first — it is the setting with the largest
effect and the least risk.

## Rules that never fire

Check the underlying detections exist at all. A rule that requires two classes
fires only where *both* are detected in the same window, so a rule can be
correct and still never match because one of its inputs is silent. The report
shows which detections were present at each kept moment.

## Where the rules live

Composition rules are yours, kept outside the repository, and their names are
your vocabulary — the report reads them back from your run rather than knowing
any of them in advance. Editing them changes nothing about the detectors; the
engine re-runs over detections that are already cached, so trying a different
rule is cheap.

See the composition engine section of `docs/DETECTION-GUIDE.md` for the rule
format itself.

## When the transcript describes something no class covers

With a transcript, the report can compare two lists: what people talked about
and what the detector produced. A word said far more often in one stretch than
in the rest of the video, matching no class name, is a gap — something the run
had no way to check.

This is worth treating differently from every other finding here, because it is
not about a setting. A weight can be raised; a class the detector does not have
cannot be conjured by any configuration. The three honest outcomes are:

- **The word describes an arrangement of classes you already detect.** Then a
  composition rule closes the gap, and the next run can check the claim. Nothing
  needs retraining.
- **The word describes something no class covers.** Then no rule helps, and the
  options are a custom category taught from examples, or accepting that this
  claim is outside what the tool can see.
- **The word is not about the picture at all.** People discuss the past, the
  weather, and each other. Most gaps are this, and closing them is wasted work.

Deciding which of the three applies is a judgement about meaning, and this tool
does not make it. The advisor can draft a rule for the first case, but only from
classes your detector actually produced in that file — a rule naming anything
else parses, loads and fires on nothing, costing a re-run to discover.

## Claims left waiting on a rule

When a rule is added to check something that was said, the report remembers the
pairing and reports the outcome on the next run. Three states, and they are not
equivalent:

- **Fired.** The arrangement occurred. That is a measurement; what it means for
  the claim is still a reading, and a rule can fire on the right arrangement of
  the wrong boxes. Watch the seconds it marked.
- **Never fired.** Weak evidence against, not proof. The arrangement may have
  happened outside the detector's view, or the rule may be too strict. Loosen it
  once before concluding anything.
- **Not in the rules.** The rule is not being evaluated, so the run says nothing
  either way. Usually this means the detection pass came from cache — the
  composition engine only runs when objects are actually re-detected.
