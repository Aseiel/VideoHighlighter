# Negative scenarios

One page per mistake. Each page says:

- what happens;
- whether and how it causes overfitting (the model learning the source
  video, its scene, people, light and camera, instead of the action);
- how to notice it;
- what was measured;
- what to do instead, linked to `../dataset-positive.md`.

## Cause overfitting, or hide it

| page | in one line |
|---|---|
| `leaky-validation.md` | validation shares videos with training, so memorising scores well |
| `epoch-chosen-on-leaky-score.md` | a leaky score picks the most overfitted checkpoint |
| `wrong-source-video-rule.md` | one video counted as many, so every split leaks |
| `training-whole-network.md` | too many free weights for too few videos |
| `frames-from-clip-start.md` | the action is not in the frames, so the scene is learned |
| `one-video-floods-dataset.md` | one video becomes what the model learns |
| `classes-from-too-few-videos.md` | a class that is really one or two scenes |
| `forced-best-guess-sorting.md` | a sort's mistakes are trained into the next model |
| `clusters-are-videos.md` | naming a group of crops that is really one video |
| `new-actions-one-video.md` | an action that exists in one video only |

## Other mistakes: lower accuracy or a misleading result, not overfitting

| page | in one line |
|---|---|
| `judging-added-video-by-overall-score.md` | a good change looks bad on the overall line |
| `wrong-encoder-input.md` | the encoder gets input it was not trained on |
| `second-crop.md` | cropping the cropper's clips again |
| `untaught-combinations.md` | a pair of actions that was never shown together |
| `typed-names-without-examples.md` | a typed name for something the encoder has no words for |
| `sorting-whole-frames.md` | sorting whole frames when the dataset is crops |
| `small-encoder-near-variants.md` | a small encoder merging near-variant classes in a sort |
