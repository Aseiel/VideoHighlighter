"""modules/teach — teach the app to find something new, end to end.

A person, or an LLM acting for one, says what to find and hands over a few
videos. This package turns that into a trained model, reusing what the app
already has at each step:

    add-class / add-video          project.py      what to find, where to look
    cut                            cut.py          fixed-length samples (ffmpeg)
    focus (actions, optional)      focus.py        the person-focused cropper,
                                                   modules/crop/actions.py
    sort                           sort.py         CLIP scores every sample
                                                   against every class, and the
                                                   project's own model does
                                                   from round 2; folders like
                                                   sorter.py's
    review                         review.py       a contact sheet and a verdict
                                                   per sample, or the folders
                                                   moved around by hand
    boxes (objects)                boxes.py        proposed boxes to accept, or
                                                   tools/labeler.py's points
    build / train                  build.py        the layouts model_training
                                   train.py        and training/ already read

Every step reads and writes files under the project folder, is safe to re-run,
and reports in JSON, so it can be driven by a script or an agent as easily as
by a person. ``status.py`` says what the next step is, with the command for
it; that one call is how an agent finds its way around.

Content-neutral, as CLAUDE.md requires: the classes are the user's data,
defined at runtime. Nothing here names or presets any subject matter.

No Qt anywhere in the package. Heavy libraries (cv2, torch, CLIP) are imported
inside the functions that need them, so the orchestration is testable without
them.
"""
