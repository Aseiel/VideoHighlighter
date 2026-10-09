# Training a whole network on few source videos

**Negative scenario.** Every weight of a large network is trained on a
dataset with a small number of source videos.

**How it causes overfitting:** the network has far more freedom than the data
has variety.
- Remembering each video's scene, people and light is easier than learning
  the action.
- So that is what tens of millions of free weights learn from about 135
  scenes.

**How you notice:** training accuracy far above held-out accuracy (e.g. 95 %
against 37 %).

**Measured** (same 29 unseen videos,
`docs/plans/2026-10-08-overfitting-and-siglip2-fine-tune.md`):

| R3D-18 | unseen videos |
|---|---|
| every layer trained | 0.37 (95 % on training clips) |
| frozen, only a small head trained | 0.45 |

Training more of the network made it worse.

**Instead:**
- Start with a frozen encoder and a small head.
- Then train only the top blocks from that head, judged on unseen videos.
- SigLIP2 with its top 4 of 12 blocks trained this way gained 7 points on
  5 folds (`../dataset-positive.md`, Training 5).
