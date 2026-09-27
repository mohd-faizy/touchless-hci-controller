"""
Configuration module for the Invisible Hand Gesture Mouse Controller.
Contains all centralized tuning parameters for camera capture, hand tracking,
gesture detection, mouse control, and HUD visualization.
"""

from typing import Tuple

# ==============================================================================
# CAMERA CONFIGURATION
# ==============================================================================
# Webcam device index (default: 0 for built-in camera, 1 or 2 for external USB webcams)
CAMERA_INDEX: int = 0

# Desired capture dimensions
FRAME_WIDTH: int = 1280
FRAME_HEIGHT: int = 720

# Target frame rate for video capture
TARGET_FPS: int = 60

# Mirror the camera horizontally so left/right movements feel intuitive
MIRROR_FEED: bool = True


# ==============================================================================
# MEDIAPIPE HAND TRACKER CONFIGURATION
# ==============================================================================
# Maximum number of hands to detect concurrently (1 ensures focused single-hand control)
MAX_NUM_HANDS: int = 1

# Minimum confidence value from hand detection model ([0.0, 1.0])
MIN_DETECTION_CONFIDENCE: float = 0.70

# Minimum confidence value from landmark tracking model ([0.0, 1.0])
MIN_TRACKING_CONFIDENCE: float = 0.70

# Handedness preference ("Right", "Left", or "Any")
TRACK_HAND: str = "Any"


# ==============================================================================
# SCREEN & COORDINATE MAPPING CONFIGURATION
# ==============================================================================
# Margins (in pixels) inside the camera frame defining the active interaction zone.
# When index finger reaches margin boundary, the cursor reaches the screen edge.
CAMERA_MARGIN_X: int = 120
CAMERA_MARGIN_Y: int = 90

# Exponential smoothing configuration
# Baseline alpha factor:
SMOOTHING_ALPHA: float = 0.25

# Dynamic adaptive smoothing (1€ Filter principles):
# Low alpha for precision & eliminating jitter when moving slowly or aiming:
SMOOTHING_ALPHA_MIN: float = 0.12
# High alpha for instant zero-lag response during rapid sweeps:
SMOOTHING_ALPHA_MAX: float = 0.55

# Sub-pixel cursor deadzone in screen pixels to completely lock jitter when holding still:
CURSOR_DEADZONE: float = 1.2

# Minimum movement delta in pixels before cursor moves:
MIN_CURSOR_DELTA: float = 0.5

# Minimum frames to confirm transition from active gesture to IDLE (prevents single-frame dropouts):
GESTURE_DEBOUNCE_FRAMES: int = 2


# ==============================================================================
# PINCH & CLICK DETECTION CONFIGURATION
# ==============================================================================
# Euclidean pixel distance threshold between thumb tip and index tip to trigger pinch
PINCH_THRESHOLD: float = 38.0

# Hysteresis release threshold: pinch releases only when distance exceeds this value.
# Prevents rapid flickering and accidental double-clicks near the threshold boundary.
PINCH_RELEASE_THRESHOLD: float = 48.0

# Debounce cooldown in seconds: minimum delay between consecutive left clicks
CLICK_COOLDOWN: float = 0.40


# ==============================================================================
# SCROLL CONFIGURATION
# ==============================================================================
# Multiplier determining how fast the screen scrolls per unit of vertical hand movement
SCROLL_SENSITIVITY: float = 2.4

# Minimum vertical hand displacement (in pixels) required to trigger a scroll event
# Prevents micro-movements from causing continuous drift when holding hand still
SCROLL_DEADZONE: float = 4.0

# Maximum scroll units allowed in a single frame to prevent uncontrollable jumping
SCROLL_SPEED_CAP: int = 25

# Exponential smoothing applied to scroll deltas for silky-smooth motion
SCROLL_SMOOTHING: float = 0.40


# ==============================================================================
# USER INTERFACE & HUD VISUALIZATION
# ==============================================================================
# Toggle visual elements on the OpenCV feed
SHOW_LANDMARKS: bool = True
SHOW_SKELETON: bool = True
SHOW_INTERACTION_ZONE: bool = True
SHOW_HUD_PANEL: bool = True
SHOW_FPS: bool = True
SHOW_PINCH_GAUGE: bool = True

# HUD Color Palette (OpenCV BGR Format)
COLOR_MOVE: Tuple[int, int, int] = (0, 230, 115)       # Vibrant Emerald Green
COLOR_SCROLL: Tuple[int, int, int] = (0, 215, 255)     # Glowing Amber / Gold
COLOR_CLICK: Tuple[int, int, int] = (50, 70, 255)      # Electric Coral Red
COLOR_IDLE: Tuple[int, int, int] = (160, 160, 175)     # Sleek Muted Slate
COLOR_ZONE: Tuple[int, int, int] = (255, 120, 200)     # Neon Violet / Purple
COLOR_BG: Tuple[int, int, int] = (25, 25, 30)          # Dark Charcoal HUD Panel
COLOR_TEXT_WHITE: Tuple[int, int, int] = (245, 245, 245)
COLOR_TEXT_DIM: Tuple[int, int, int] = (140, 140, 150)
COLOR_ACCENT: Tuple[int, int, int] = (255, 200, 50)    # Cyan / Electric Blue
COLOR_REC: Tuple[int, int, int] = (0, 0, 255)          # Bright Red for Recording Indicator


# ==============================================================================
# CONTROLS & SAFETY KEYS
# ==============================================================================
# Key codes to terminate application cleanly
EXIT_KEYS = [ord('q'), ord('Q'), 27]  # 'q', 'Q', or ESC key

# Key codes to toggle pause mode (temporarily freeze mouse control)
PAUSE_KEYS = [ord('p'), ord('P')]

# Key codes to toggle live demo video recording
RECORD_KEYS = [ord('r'), ord('R')]

# Key codes to capture an instant PNG snapshot
SNAPSHOT_KEYS = [ord('s'), ord('S')]

# Recording defaults
DEFAULT_VIDEO_PATH: str = "assets/demo.mp4"
DEFAULT_SNAPSHOT_PATH: str = "assets/sample_project.png"

