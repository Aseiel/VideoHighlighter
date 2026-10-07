# Actions by name: 4 or 8 frames per window

**Question:** SigLIP2 matches actions by name on 5-second windows (one every
2.5 s), reading 4 frames of each. That is about 1.6 analysed frames a second,
where the Intel/R3D pass analysed every 5th frame. Does reading 8 frames per
window find more, or find better?

**Answer, on what was measured:** 8 frames takes 1.5x as long, finds nothing
that 4 frames misses, and on the one action checked by eye it dropped a few
windows that only looked like the action. Both are offered as the
**Frames per window** setting (Advanced → Action Recognition); 4 is the
default.

| Setting | Speed | What it does |
|---|---|---|
| 4 frames | faster | Finds the most; a few windows only look like the action |
| 8 frames | about 1.5x slower | Finds a subset of 4's results; the dropped ones were false matches, on the action checked |

A model trained on your own clips is not affected: it reads the number of
frames it was trained on, whatever the setting says.

## The measurement

One video: a 23-minute TV episode, 640x360, 24.9 fps, 554 windows. Arc A750,
OpenVINO GPU for the encoder, YOLOX for the people boxes. The same windows and
the same rule (typed: among the window's 3 strongest names with 5 %+ of the
share; blank: the strongest name at 35 %+) at both settings.

| | 4 frames | 8 frames |
|---|---|---|
| Run time (whole pass) | 21 s | 31 s |
| Blank field: seconds tagged | 494 (17 actions) | 454 (same top actions) |
| "punching person (boxing)" | 30 s | 20 s, all inside 4's 30 |
| "slapping" | 94 s | 91 s (64 s agree; ~30 s each way differ) |
| "sword fighting" | 40 s | 29 s (20 s agree) |
| "dancing" | 524 s | 533 s |
| "high kick", "dancing ballet" (not in the video) | 0 | 0 |

**Punching, checked by eye.** Both found the fighting stance at 18:11-18:13
and a clenched-fist gesture at 0:57-1:07 (loose: talking, not punching).
Only 4 frames also reported 5:48-5:50 (two people talking, a robot behind
them), 18:15 (someone crouching after the fight) and 18:18 (a transformation
effect): none of them punching. 8 frames found nothing 4 frames missed, so
the worry behind the test (short actions falling between sampled frames) did
not show up here.

**Not measured:** the "slapping" and "sword fighting" differences were not
checked by eye, so whether 8 frames is stricter in general, or only on
punching, is open.

## Why no accuracy percentage

A percentage needs the true answer for every window: where each action really
happens. That has not been marked for any video yet, so any figure here would
be invented. To get one: mark the real stretches of a few actions in a few
minutes of 2-3 videos, then score both settings against them (precision: how
much of what each reports is right; recall: how much of what is there each
finds).
