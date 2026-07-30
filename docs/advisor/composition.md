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
