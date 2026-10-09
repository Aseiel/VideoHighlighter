# Classes with too few examples or from too few videos

**Negative scenario.** A class's clips all come from one or two source
videos, or there are only a handful of them.

**How it causes overfitting:** the class *is* those videos.
- If every example shares one scene, the model has no way to tell the
  action from the scene, and learns both together.
- On new footage, anything resembling that scene is pulled into the class.

**How you notice:**
- A small class gets many wrong guesses on new footage.
- It never reaches a trust threshold.

**Measured:**
- Classes under ~20 clips attracted many wrong guesses on new footage.
- Fewer than 5 single-action clips: the trainer leaves the class out.
- Hits from fewer than 3 videos: it can never be trusted, only suggested.

**More clips of a video already in the set do not fix this; more videos do.**

**Instead:** get every class into 3+ source videos (`../dataset-positive.md`,
Building 1-2).
