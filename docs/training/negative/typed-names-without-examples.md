# Expecting a typed name to find what image-text data rarely shows

**Negative scenario.** An action is searched for by typing its name, with no
examples, when it is the kind of content web image-text data covers poorly.

**Not overfitting, but a model that has nothing to match.** Matching by name
works only for what the encoder learned names for.

**How you notice:** results no better than guessing. Everyday actions work;
specialised ones do not.

**Measured:** typing this dataset's own class names with no training scored
0.10-0.16, where "always the largest class" scores 0.14. A taught head on
the same encoder: 0.53-0.58
(`docs/plans/2026-10-01-action-models-measured.md`).

**Instead:** use the typed name to start and to search, then teach from
examples.
