"""
Mouse Controller module for the Invisible Hand Gesture Mouse Controller.
Encapsulates high-performance OS cursor automation, coordinate space transformation,
adaptive velocity-based exponential smoothing (1€ Filter principles),
sub-pixel jitter deadbanding, and native Windows API acceleration.
"""

import sys
import ctypes
from typing import Tuple, Optional
import numpy as np
import pyautogui

import config

IS_WINDOWS = sys.platform == "win32"


class MouseController:
    """
    Translates camera-space coordinates into silky-smooth, high-precision screen actions.
    """

    def __init__(
        self,
        smoothing_alpha: float = config.SMOOTHING_ALPHA,
        margin_x: int = config.CAMERA_MARGIN_X,
        margin_y: int = config.CAMERA_MARGIN_Y,
        min_cursor_delta: float = 0.5,
        alpha_min: float = config.SMOOTHING_ALPHA_MIN,
        alpha_max: float = config.SMOOTHING_ALPHA_MAX,
        cursor_deadzone: float = config.CURSOR_DEADZONE,
    ) -> None:
        """
        Initializes the mouse controller with screen dimensions and adaptive filter settings.

        Args:
            smoothing_alpha: Baseline filter coefficient.
            margin_x: Horizontal camera padding for the active interaction zone.
            margin_y: Vertical camera padding for the active interaction zone.
            min_cursor_delta: Minimum delta (pixels) required to trigger moveTo.
            alpha_min: Dynamic alpha floor for micro-movements and precision pointing.
            alpha_max: Dynamic alpha ceiling for rapid screen sweeps.
            cursor_deadzone: Screen-space pixel deadzone to completely eliminate resting jitter.
        """
        self.smoothing_alpha = float(smoothing_alpha)
        self.margin_x = int(margin_x)
        self.margin_y = int(margin_y)
        self.min_cursor_delta = float(min_cursor_delta)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.cursor_deadzone = float(cursor_deadzone)

        # Retrieve primary display resolution
        self.screen_width, self.screen_height = pyautogui.size()

        # PyAutoGUI performance & safety optimization
        # Setting PAUSE = 0.0 enables zero-latency real-time 60 FPS cursor motion.
        pyautogui.PAUSE = 0.0
        # Disabling fail-safe prevents unhandled exceptions when cursor moves near screen edges
        pyautogui.FAILSAFE = False

        # State tracking for exponential smoothing filter
        self.smoothed_x: Optional[float] = None
        self.smoothed_y: Optional[float] = None

        # Camera coordinate pre-filter memory
        self.cam_smoothed_x: Optional[float] = None
        self.cam_smoothed_y: Optional[float] = None

        # Tracking last dispatched screen positions
        self.last_dispatched_x: int = self.screen_width // 2
        self.last_dispatched_y: int = self.screen_height // 2

    def map_coordinates(
        self,
        cam_x: float,
        cam_y: float,
        cam_width: int,
        cam_height: int
    ) -> Tuple[float, float]:
        """
        Maps camera coordinates within the active interaction margin to the full screen.

        Applies linear interpolation and hard boundary clamping:
        [margin_x, cam_width - margin_x]  -> [0, screen_width - 1]
        [margin_y, cam_height - margin_y] -> [0, screen_height - 1]

        Args:
            cam_x: Raw X coordinate in camera space.
            cam_y: Raw Y coordinate in camera space.
            cam_width: Width of camera frame in pixels.
            cam_height: Height of camera frame in pixels.

        Returns:
            Tuple (screen_x, screen_y) clamped within display resolution.
        """
        # Linear interpolation with boundary clamping via NumPy
        screen_x = float(
            np.interp(
                cam_x,
                [self.margin_x, max(self.margin_x + 1, cam_width - self.margin_x)],
                [0, self.screen_width - 1],
            )
        )
        screen_y = float(
            np.interp(
                cam_y,
                [self.margin_y, max(self.margin_y + 1, cam_height - self.margin_y)],
                [0, self.screen_height - 1],
            )
        )

        # Hard clamp to guarantee cursor never crosses OS screen bounds
        screen_x = float(np.clip(screen_x, 0, self.screen_width - 1))
        screen_y = float(np.clip(screen_y, 0, self.screen_height - 1))

        return screen_x, screen_y

    def smooth_coordinates(
        self,
        target_x: float,
        target_y: float,
        use_adaptive: bool = False
    ) -> Tuple[float, float]:
        """
        Applies an Exponential Moving Average (EMA) low-pass filter to smooth cursor coordinates.
        Supports dynamic velocity-based adaptive alpha (1€ filter style):
        - Slow speed / small movements -> low alpha (heavy smoothing, zero jitter)
        - Fast speed / wide sweeps -> high alpha (instant response, zero lag)

        Args:
            target_x: Desired screen X coordinate.
            target_y: Desired screen Y coordinate.
            use_adaptive: Whether to dynamically scale alpha based on movement speed.

        Returns:
            Tuple (smoothed_x, smoothed_y) as floats.
        """
        if self.smoothed_x is None or self.smoothed_y is None:
            # First frame initialization: avoid interpolation jump from origin
            self.smoothed_x = target_x
            self.smoothed_y = target_y
            return self.smoothed_x, self.smoothed_y

        if use_adaptive:
            # Calculate displacement distance between target and current smoothed position
            dist = np.hypot(target_x - self.smoothed_x, target_y - self.smoothed_y)

            # Velocity-based dynamic alpha:
            # < 5px: alpha = alpha_min (strong smoothing to lock out hand tremor)
            # > 65px: alpha = alpha_max (fast response without trailing lag)
            dist_min = 5.0
            dist_max = 65.0
            ratio = np.clip((dist - dist_min) / (dist_max - dist_min), 0.0, 1.0)
            alpha = float(self.alpha_min + ratio * (self.alpha_max - self.alpha_min))
        else:
            alpha = self.smoothing_alpha

        self.smoothed_x = alpha * target_x + (1.0 - alpha) * self.smoothed_x
        self.smoothed_y = alpha * target_y + (1.0 - alpha) * self.smoothed_y

        return self.smoothed_x, self.smoothed_y

    def move(
        self,
        cam_x: float,
        cam_y: float,
        cam_width: int,
        cam_height: int
    ) -> Tuple[int, int]:
        """
        Executes real-time smooth cursor movement based on camera landmark position.
        Uses adaptive filtering, sub-pixel deadbanding, and native OS APIs.

        Args:
            cam_x: Landmark X coordinate in camera space.
            cam_y: Landmark Y coordinate in camera space.
            cam_width: Camera frame pixel width.
            cam_height: Camera frame pixel height.

        Returns:
            Tuple (final_screen_x, final_screen_y) applied to the system cursor.
        """
        # Step 1: Pre-filter camera input coordinates to remove sensor quantization noise
        if self.cam_smoothed_x is None or self.cam_smoothed_y is None:
            self.cam_smoothed_x = cam_x
            self.cam_smoothed_y = cam_y
        else:
            # Light 0.60 EMA on raw pixels before monitor resolution amplification
            self.cam_smoothed_x = 0.60 * cam_x + 0.40 * self.cam_smoothed_x
            self.cam_smoothed_y = 0.60 * cam_y + 0.40 * self.cam_smoothed_y

        # Step 2: Map camera coordinates to screen dimensions
        target_screen_x, target_screen_y = self.map_coordinates(
            self.cam_smoothed_x, self.cam_smoothed_y, cam_width, cam_height
        )

        # Step 3: Apply adaptive velocity-based exponential smoothing
        filtered_x, filtered_y = self.smooth_coordinates(
            target_screen_x, target_screen_y, use_adaptive=True
        )

        final_x = int(round(filtered_x))
        final_y = int(round(filtered_y))

        # Step 4: Sub-pixel deadband check
        # Only dispatch OS call if displacement exceeds deadzone, preventing 1-pixel micro-jitter
        delta = np.hypot(filtered_x - self.last_dispatched_x, filtered_y - self.last_dispatched_y)
        if delta >= self.cursor_deadzone:
            # Native Windows SetCursorPos has 0.005ms latency and zero overhead
            if IS_WINDOWS:
                try:
                    ctypes.windll.user32.SetCursorPos(final_x, final_y)
                    self.last_dispatched_x = final_x
                    self.last_dispatched_y = final_y
                except Exception:
                    pyautogui.moveTo(final_x, final_y)
                    self.last_dispatched_x = final_x
                    self.last_dispatched_y = final_y
            else:
                try:
                    pyautogui.moveTo(final_x, final_y)
                    self.last_dispatched_x = final_x
                    self.last_dispatched_y = final_y
                except Exception:
                    pass

        return final_x, final_y

    def click(self) -> None:
        """Triggers a single debounced OS left mouse click."""
        try:
            pyautogui.click()
        except Exception:
            pass

    def scroll(self, amount: float) -> None:
        """
        Scrolls the screen by the given amount.
        Positive amount scrolls up; negative amount scrolls down.

        Args:
            amount: Scroll delta clicks to apply.
        """
        if abs(amount) < 0.5:
            return

        scroll_clicks = int(round(amount))
        try:
            pyautogui.scroll(scroll_clicks)
        except Exception:
            pass

    def reset(self) -> None:
        """
        Resets smoothed coordinates to prevent jump artifacts when
        hand re-enters camera frame after being absent.
        """
        self.smoothed_x = None
        self.smoothed_y = None
        self.cam_smoothed_x = None
        self.cam_smoothed_y = None
