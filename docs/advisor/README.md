# Advisor knowledge base

These pages are what the advisor knows. `modules/highlight_advice.py` decides
*which* page applies by computing findings from the report; these files supply
the explanation and the remedy in language a user can act on.

**The knowledge lives here, not in a model.** Nothing was trained on this
folder. The advisor reads it at run time, which means a wrong or outdated
answer is fixed by editing a Markdown file, not by collecting data and
retraining. It also bounds what the advisor can claim: it has no material
beyond these pages plus the numbers in the report.

## Files

| topic | when the advisor reaches for it |
|---|---|
| `weights.md` | the signal weights produced a cut the user did not want |
| `thresholds.md` | a detector is on but found nothing, or the cut is short |
| `coverage.md` | the cut is drawn from one stretch of the video |
| `variety.md` | every clip looks the same |
| `composition.md` | raw detections vs. composed events |
| `training.md` | the detector cannot see it at any threshold |
| `measuring.md` | something was said that no signal in the run measured |

## Writing rules

- **A topic answers one question**, the one the finding raises. Anything else
  belongs in `docs/DETECTION-GUIDE.md`, which these pages link to rather than
  restate.
- **Describe mechanisms, never subject matter.** Same rule as the rest of the
  repo: "the class you trained", "the category you taught", never an example of
  what that class might be. Users' categories are their data.
- **Prefer a number to an adjective.** "Below about 0.25 confidence you get
  false positives" is actionable; "a low threshold" is not.
- **Say when the answer is 'this cannot work'.** A label-based detector has no
  class for what it was never given; no setting fixes that, and pretending
  otherwise sends users round a loop. Point at `training.md`.
