# Community models

Design note. Kept identical in both repos.

## The bet

Commercial highlight tools add support one domain at a time — a handful of
titles at launch, a few dozen a year later — because every one is a model
somebody on staff tuned. That does not scale to the long tail: nobody on a
payroll is going to build the detector for one niche hobby, one sport's
scoreboard, one game's kill-feed.

A community can, because every one of those niches has somebody who cares about
it. The goal is **thousands of small models, each made by someone who wanted
it**, found on the site under a category, and one click away from working in
the app.

Most people will never "train a model" on purpose. So the app never asks them
to. It asks them to point at the thing they want found, checks a few guesses
with them, and — once it works for them — asks politely whether they'd like to
share it.

## Two kinds of model

| kind | what it is | cost to make | good for |
|---|---|---|---|
| **prototype** | a taught category: the average CLIP vector of a few examples (`llm/clip_categories.py`) | seconds, no GPU | "find more moments like these" in similar footage |
| **detector** | a YOLOX detector (`training/train_yolox_run.py`) | a few minutes of checking + GPU minutes | fitted boxes, fast, counts, composition rules |

The prototype is the on-ramp. Anyone can make one in the time it takes to mark
three frames, which is what gets people to share their *first* model. The
detector is where the ones who care go next.

## What is built

- **Detection is YOLOX in both editions** (`modules/vision/detection_backend.py`,
  `modules/vision/yolox_models.py`). No AGPL detector anywhere the app runs, so no
  model trained here inherits a licence that stops it being shared.
- **Training in the app** — the Train tab (`modules/ui/training_panel.py`) over
  `modules/vision/label_store.py` → `training/train_yolox_run.py` →
  `training/export_yolox.py`. Progress, cancel, any device.
- **Time and progress** — an estimate before the run, measured on this
  computer after the first one (`training/train_estimate.py`); stages, time left,
  and after each round what the model finds on held-out frames, with a
  "Watch it learn" window (`modules/vision/training_preview.py`).
- **The hub** — `model_hub/` (tests: `tests/test_model_hub.py`):
  - *Package*: a folder of `model.onnx`, `videohighlighter.json` (manifest with
    task, labels, input format, category, measured metrics, checklist, sha256),
    a generated `README.md` model card, optional `LICENSE`. Nothing else, checked
    before upload, after download and before use, with one test inference.
  - *Share*: after a finished run the Train tab asks, politely, whether to share.
    The wizard arrives filled in from training; the person writes a name and a
    description, picks a category and confirms the checklist.
    `hub.release_the_kraken` re-checks everything, then `hub.publish` uploads to
    the author's own Hugging Face account, tagged `videohighlighter` and
    `vh-cat-…` per category level.
  - *Install*: Advanced → *Community models…* lists the catalog by category,
    installs a model pinned to one commit into `<app data>/models/community/`,
    and it appears in the object model list.
  - *Site*: `models.html` on the site reads the public Hugging Face API — no
    server — with the category tree, measured results and a report link.

## What is next, in order

1. **Draw boxes on the video** instead of importing from the labeller, with
   tracking so one box becomes many labels. Until then only people who use
   `tools/labeler.py` can train at all — the biggest gap left.
2. **One real publish/install round trip** with a test Hugging Face account.
   Everything network-facing is tested offline only.
3. **Signing in without a pasted token.** Today sharing needs a Hugging Face
   account and a write token — real friction. "Sign in with Hugging Face"
   (OAuth) or a no-account upload into a reviewed organisation would remove it.
4. **Suggested category.** Rank the categories already in use against the
   model's class names; pre-select the best. Still no category list in code.
5. **Taught categories as packages.** The 30-second CLIP prototype has no task
   in the manifest yet; it is the on-ramp and should become shareable.
6. **Improve someone else's model.** Install, add your footage, retrain, share
   as a new version with the original credited.
7. **`videohighlighter://install?id=…`** so the site's cards install in one click.

## Nudging people toward making one

- **Never say "train".** Say *teach it what to look for, check its guesses, and
  it gets better each time* (see `docs/CUSTOM-MODEL-TRAINING.md`).
- **Offer at the moment of success.** The share button appears when a model
  has just worked for the person — not in a menu they have to find.
- **Show the honest number.** "Found it in 9 of 10 places you checked" is what
  makes a shared model trustworthy and what tells the next person whether it
  will work for them.
- **Credit by name.** A display name on the card and on the site is most of the
  motivation anyone needs. It is optional.
- **Seed it.** The first twenty models on the site are ours. An empty gallery
  asks nobody to contribute.

## The rules that keep it safe to run

*Not legal advice — the working rules this design is built on. Have a
copyright lawyer read this before the upload service goes live.*

**Share the model, never the material.** Training on footage and redistributing
that footage are different acts. In the EU, text-and-data-mining exceptions
(DSM Directive art. 3 and 4, implemented nationally) generally cover analysing
lawfully accessed content, including commercially, unless the rightsholder has
opted out in a machine-readable way. They do not cover handing the frames,
clips or audio to other people. A small detector's weights or a category's
average vector hold no recognisable copy of the footage; a thumbnail would.
So the package format has no place for media at all, and verification rejects
anything outside an exact list of files.

**The person sharing vouches for it.** Four checklist items, confirmed in the
wizard and carried in the manifest and on the model card: they checked the
source's terms, trained on footage they may use, ship no training data, and the
model does not generate content. Nothing uploads without all four.

**Permissive licences only.** Apache-2.0, MIT, CC-BY-4.0, CC0-1.0. A model that
restricts who may use it is not really shared.

**Names describe; they do not claim.** A title may be named to say what a model
is for ("events in <title>"). No logos, no artwork, nothing suggesting the
publisher endorses it.

**Every published model can be taken down.** A visible report link on every
card, a contact address, and a fast response. Hosting on a platform that
already runs notice-and-takedown (Hugging Face) is part of why it is the
default — and the model sits in its author's account, so they answer for it.

**We can delist without waiting for anyone.** A model tagged `videohighlighter`
is listed as soon as it is published, unreviewed. `models-blocklist.json` on the
site names repositories the app and the site must not show; both read it on
every catalog load, and `hub.install` refuses a blocked repository. Content
neutrality in the codebase does not mean the listing shows anything: a report
leads to a delisting in minutes, by editing one file. Reviewed models are marked
Verified (`VERIFIED_AUTHORS`) and listed first.

**Game rules are their own question.** Publishers' content-creator and
video policies decide what may be recorded and posted; some address AI. A
model that recognises on-screen events from a player's own recordings is a
different thing from redistributing the game's assets, but the policy of the
title still applies to the person making it.

## How others get per-title support

Worth knowing, not copying. Game-clip products tend to get per-title events one
of two ways: from game-event APIs published through partnerships (Overwolf's
game-events service is the common one), or from their own vision models trained
on gameplay. The second kind generally do not publish their training data, and
lean on publishers' content-creator policies and on the fact that publishers
want more clips of their games in circulation. Whether any of them hold
individual licences from publishers is not public, and this design does not
assume one is needed for a model that ships no game assets — which is exactly
what the package format guarantees.
