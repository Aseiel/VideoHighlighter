# Training knowledge base

What is known, by measurement, about building a dataset and training an
action model on it, written so that the advisor, an agent running
`modules/teach`, or a person sorting clips can act on it.

The pages:

| page | read it when |
|---|---|
| `dataset-positive.md` | building, growing or auto-sorting a dataset: what to do, as one checklist |
| `positive/` | one small page per change that measurably improved accuracy, with its number and how solid it is (index in `positive/README.md`; start with `positive/measure-on-unseen-videos.md`) |
| `negative/` | checking a dataset, a sort or a training run: one small page per mistake, what it does (most cause or hide overfitting), how it shows up, and the fix (index in `negative/README.md`) |

Every rule here comes from a measurement in `docs/plans/`, linked from the
rule. The numbers are from one hand-sorted dataset (2,711 clips from 135
source videos, cut by the app's cropper), so treat them as sizes, not
guarantees.

## Words used here

- **Source video:** the video a clip was cut from. In the app's naming it is
  the clip name before the first `_temp` / `_highlight`
  (`modules/teach/benchmark.py`). Clips from one source video share scene,
  people, light and camera, so they are near-copies, not independent
  examples.
- **Held-out / unseen:** scored on source videos the model never trained on,
  not even on other clips from them. This is the situation of every new
  video a user analyses, so it is the only score that predicts real use.
- **Leaky:** scored on clips whose source video is also in training. Leaky
  scores are always higher and reward remembering the scene.
- **Trusted:** a class reported only above the score where its held-out
  precision is reliably high enough (`model_training/action_head/trust.py`).

## Writing rules

Same as `docs/advisor/README.md`:

- **Mechanisms, never subject matter.** "a rare class", "a near-variant of a
  large class", never what a class shows.
- **A number before an adjective.**
- **Say when something cannot work.**
