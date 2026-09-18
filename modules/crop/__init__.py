"""
modules/crop — the cropping subsystem.

Two objectives share one set of primitives:

    focus  (actions.py)  "where is the action?"      -> frame toward it
    avoid  (avoid.py)    "who must NOT be on screen?" -> frame away from them

and both render through core.py, which is pure geometry plus the letterbox/pad
step and knows about neither.

    core     box math, smoothing, padding, safe cropping  (shared by both)
    config   the tuning surface
    pose     keypoint estimation — dormant, see its docstring
    people   how many people are in this video
    zones    where the action is -> crop count and positions
    track    following those crops frame to frame
    debug    visualisation, no policy
    actions  focus orchestration + batch CLI
    avoid    exclusion cropper, independent of the above

Split out of a single 3,899-line crop_actions.py. Kept byte-identical with the
free edition — see the two-repo split in CLAUDE.md.
"""
