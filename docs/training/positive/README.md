# Positive scenarios

One page per change that measurably improved accuracy. Each page says:

- what to do;
- why it helps;
- what was measured;
- how solid that measurement is (seeds, splits, one video or many);
- where it lives in the code, and the negative scenario it answers.

`../dataset-positive.md` is the same knowledge as one checklist; these pages
are the evidence behind it.

## The first rule

| page | in one line |
|---|---|
| `measure-on-unseen-videos.md` | without a score on unseen videos, every "it will improve" is a guess |

## Training

| page | gain |
|---|---|
| `stronger-pretrained-encoder.md` | Intel 0.42 → SigLIP2 base 0.555 (the largest) |
| `frozen-encoder-first.md` | R3D-18 0.37 → 0.45, frozen instead of fully trained |
| `partial-fine-tune.md` | SigLIP2 0.533 → 0.603 on 5 folds; sorted with confidence 44 % → 57 %; on a new video 1.8× the labels found |
| `framing-and-frames-across-clip.md` | Intel 0.348 → 0.418, with the colour fix |
| `encoder-input-as-declared.md` | Intel +3-4, colours as the IR declares |
| `softened-class-weights.md` | +3.0, class weights at power 0.5 |
| `small-head-on-small-data.md` | +2, a linear layer instead of an MLP |
| `teaching-pairs.md` | both actions of a pair found 42 % instead of 0-2 % |

## Dataset and sorting

| page | gain |
|---|---|
| `combined-encoders-for-sorting.md` | 0.522 → 0.627; a movement-defined class 0.12 → 0.48 |
| `trust-thresholds.md` | 66 % sorted at 77 % precision instead of forced guesses |
| `capped-reviewed-footage.md` | thin actions +7-10 recall, others protected |
| `action-vs-no-action-probe.md` | empty crops: AUC 0.66 → 0.80 |
| `two-views-at-analysis.md` | +1.7 at twice the encoder time |
