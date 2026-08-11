# Something was said that nothing measured

The report can quote a line and have nothing to say about it. That is not a
failure of the settings — it is the run reaching the edge of what it was able
to check at all. No weight, threshold or rule changes it, because there is no
signal to weight.

The question this page answers is the next one: **what would it take to
measure that, and is it worth it?**

## Rule out the free answer first

Before spending anything, find out whether the thing is an *arrangement* of
classes the detector already produces — one inside another, several at once,
none of them present. If it is, a composition rule expresses it exactly, costs
minutes to write, and needs no new model.

Ask the advisor to draft the rule ("Check something that was said…"). It
replies that the claim cannot be built from the available classes when it
cannot, and that reply is worth having: it costs one model call and closes off
the cheapest route or opens it.

Rules test *state*, not onset, and a new rule cannot fire until object
detection runs again — a cached detection pass skips the composition engine.

## The two that are usually the real choice

### Fastest — search the video for the words that were said

The claim came out of a transcript, so you already have the wording. Type it
into CLIP search and see which moments come back. One pass embeds the video;
after that a query is arithmetic on a small array and costs nothing, so trying
five phrasings is as cheap as trying one.

- **Right when** the thing can be put into plain words, and something said out
  loud on camera usually can.
- **Wrong when** it is small in the frame. CLIP scores *scenes*, so a target
  that occupies a few percent of the picture is drowned by everything around
  it — you end up asking "does this look like the scene where that happens".
- **Judge it by ranking, not by the number.** The score is calibrated against
  generic negatives, which makes a threshold transferable between videos, but
  the ordering is what tells you whether it found the thing.
- **You get a score, not a box.** No counting, and no way to express "none of
  it is here".

Try this first in almost every case. It fails cheaply, and the failure is
informative — see the control test below.

### Most reliable — train a class of your own

Boxes at frame rate: countable, usable in composition rules, and reusable on
every video you analyse afterwards. It costs a session and a GPU.

Worth it when the thing matters enough to justify labelling, when you need
boxes rather than scores, or when search has already been tried and could not
separate the moments you wanted from the ones you did not.

The part people get wrong is the dataset, not the training. Collect frames
from the conditions it *fails* in rather than more of what already works;
budget roughly a third of the set for frames containing whatever it will
confuse for the target, with nothing boxed at all. Diversity beats volume.

## The five-minute test that decides between them

Search is cheap enough to be a measurement of the *model* rather than of the
video, and that is the most useful thing you can do with it before committing
an afternoon. Run your real phrase and a control phrase — something ordinary
you know is in the shot — over the same index:

```bash
python -m llm.clip_index --video "your.mp4" --interval 2 --query "your thing" --query "a close-up" --topk 10
```

- Control finds its moments, yours does not → the model has no useful
  representation of your subject. No threshold rescues that; you need a
  trained class.
- Both find theirs → you have just avoided a session of labelling.
- Neither finds anything → check the setup before blaming the model.

## One more, for a particular kind of thing

**An action model** is the only engine here that sees time. Reach for it when
what was said is a *movement* — something that exists across several seconds
and is invisible in any single frame. It is fed a cropped region, so it cannot
separate two categories that differ only by *where* they happen; train one
class for the movement and split it with a composition rule.

## What not to mistake for a measurement

Scoring the seconds where the thing is *talked about* — a transcript keyword
weight — costs nothing and is worth having while you decide. It is not a
measurement of the thing. People describe what happened long after it
happened, and the thing can happen with nobody saying a word. Use it as a
bookmark, and never report it as evidence that the thing was there.

## Related

- `docs/DETECTION-GUIDE.md` — what each engine is, what it costs, and the
  measured limits behind the numbers on this page.
- `training.md` — the same choice framed from a detector that cannot see
  something, rather than from a claim nothing checked.
- `composition.md` — what a rule can and cannot express.
