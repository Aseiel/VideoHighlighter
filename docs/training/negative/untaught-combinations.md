# Expecting a combination that was never taught

**Negative scenario.** Clips showing two actions at once are expected to be
found, but the dataset never showed that pair together.

**Not overfitting, but a gap in what was taught.** The head learns the
combinations it is shown; it does not invent new ones.

**How you notice:** in a clip with two actions, only one is reported. The
pairs that work are exactly the ones that have an `a_b` folder.

**Measured:**
- Taught pairs: both actions in the top two on unseen videos about 42 % of
  the time.
- Untaught pairs: 0-4 %.

**Instead:** put clips that show both actions in a folder named `a_b`
(`../dataset-positive.md`, Building 5).
