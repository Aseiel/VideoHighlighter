"""
config.py — the tuning surface, in one place on purpose.

The original crop_actions.py opened with ~90 lines of ALL-CAPS knobs, and that
part was not the problem: a cropper is tuned by editing thresholds, and having
them in one screen is what made that possible. The split keeps that surface
intact rather than scattering each constant next to its function — module-private
values stay private, but anything a person would actually turn lives here.

NOT CURRENTLY READ BY ANYTHING (kept, not deleted — several belong to analysis
that is dormant rather than gone; see pose.py):
    STICKY_FRAMES, ACTION_LOCK_FRAMES, MAX_MISSING_FRAMES, MAX_ASPECT_RATIO,
    MIN_ASPECT_RATIO, USE_POSE_CENTERING, INTERACTION_ZONE_EXPANSION,
    PEOPLE_COUNT_CONFIDENCE_BOOST, USE_COHERENCE_DETECTION,
    COHERENCE_THRESHOLD_HIGH, COHERENCE_THRESHOLD_LOW, MIN_COHERENCE_SAMPLES,
    POSE_VALIDATION_IOU_THRESHOLD.

A NOTE ON THE CONFIDENCE CONSTANTS BELOW
    PERSON_DETECTION_CONF and friends sit below 0.40 deliberately — the comments
    explaining why ("catches partial people") are the design. They only take
    effect if the detector itself is constructed permissively; see the
    score_thr argument where YoloxPeopleDetector is built in actions.py.
"""


# ===== ENHANCED CONFIG =====
INPUT_FOLDER = "input_videos"
OUTPUT_FOLDER = "output_videos"
MIN_PEOPLE_REQUIRED = 2
MAX_PEOPLE = 3
PEOPLE_SAMPLE_FRAMES = 40

# The floor the YOLOX detector itself is built with. Everything below is a
# per-call conf= filter applied AFTER inference, so this has to sit under the
# lowest of them or they are decorative — see the comment where
# YoloxPeopleDetector is constructed in actions.py.
DETECTOR_SCORE_FLOOR = 0.05

# IMPROVED: Lower confidence thresholds for better partial person detection
PERSON_DETECTION_CONF = 0.10  # Lowered from 0.4 - catches partial people
PERSON_DETECTION_CONF_ZONES = 0.12  # Even lower for zone analysis
MIN_PERSON_AREA_RATIO = 0.0005  # Lowered from 0.003 - allows smaller people

STICKY_FRAMES = 30
SMOOTHING_WINDOW = 15
CALIBRATION_FRAMES = 30
PADDING_COLOR = (0, 0, 0)
BOX_EXPANSION = 0.20  # For good detections
ACTION_LOCK_FRAMES = 90
MAX_MISSING_FRAMES = 45

# Minimum box dimensions (as percentage of frame)
MAX_ASPECT_RATIO = 1.4
MIN_ASPECT_RATIO = 0.6

# ROI Detection Settings
USE_ROI_DETECTION = True
USE_POSE_FOR_ROI = True
ROI_CONFIDENCE_THRESHOLD = 0.15
MIN_POSE_KEYPOINTS = 2
ROI_SMOOTHING = True
ROI_SMOOTH_WINDOW = 8

# Pose Settings
USE_POSE_ESTIMATION = True
USE_POSE_CENTERING = False
POSE_CONFIDENCE_THRESHOLD = 0.3
MIN_EXPANSION = 0.1
MAX_EXPANSION = 0.15

PERSON_DETECTION_CONF_TRACKING = 0.12 # can affect window size!

# ===== DEBUG VISUALIZATION CONFIG =====
DEBUG_MODE = True  # Set to True to enable debug visualization
DEBUG_SAMPLES = 4  # Number of sample frames to visualize
DEBUG_OUTPUT_FOLDER = "debug_visualizations"  # Folder for debug images

# Video debug settings
DEBUG_CREATE_VIDEOS = True  # Create full debug videos
DEBUG_VIDEO_FOLDER = "debug_videos"  # Folder for debug videos
DEBUG_VIDEO_SIDE_BY_SIDE = False  # Show original + debug side-by-side
DEBUG_SHOW_METRICS = True  # Show tracking metrics overlay

# ===== ENHANCED PEOPLE DETECTION =====
# Detect partial people and interaction zones
USE_PARTIAL_PERSON_DETECTION = True
PARTIAL_PERSON_MIN_AREA_RATIO = 0.0005  # Even smaller for legs-only, torsos, etc.
INTERACTION_ZONE_EXPANSION = 0.3  # Expand detection zones to catch nearby partial people
POSE_KEYPOINT_CLUSTER_DETECTION = True  # Detect people by keypoint clusters
MIN_KEYPOINT_CLUSTER_SIZE = 2  # Minimum keypoints to count as a person
KEYPOINT_CLUSTER_RADIUS = 70  # Pixels - how close keypoints must be

# People counting adjustment
PEOPLE_COUNT_CONFIDENCE_BOOST = True  # Use multiple detection methods
COMBINE_BBOX_AND_POSE_COUNTS = True  # Merge bbox and pose detections
ADJACENCY_BONUS = True  # If 2 detected, check if 3rd is likely nearby


# COHERENCE DETECTION SETTINGS
USE_COHERENCE_DETECTION = True
COHERENCE_THRESHOLD_HIGH = 0.6  # Above this = same action
COHERENCE_THRESHOLD_LOW = 0.4   # Below this = different actions
MIN_COHERENCE_SAMPLES = 5

# Jump Resistance Settings
MAX_JUMP_RATIO = 0.18           # Max jump = 18% of frame width per frame
JUMP_RESISTANCE_MIN_HISTORY = 4  # Enforce jump limits after 4 frames of history

# ===== POSE VALIDATION FOR LOW-CONFIDENCE DETECTIONS =====
# Below this confidence, a bbox detection MUST have matching pose keypoints
# to be counted as a real person. Above this, trust the detection as-is.
POSE_VALIDATION_CONF_THRESHOLD = 0.25
POSE_VALIDATION_MIN_KEYPOINTS = 1       # Minimum keypoints inside bbox to validate
POSE_VALIDATION_KEYPOINT_CONF = 0.15    # Minimum keypoint confidence for validation
POSE_VALIDATION_IOU_THRESHOLD = 0.15    # Min IoU between bbox and pose bbox to match


# ======================================
