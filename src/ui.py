"""
On-screen user interface (HUD) module for the Hand Gesture Mouse Controller.
Renders real-time telemetry cards, interaction zone boundaries, landmark halos,
pinch indicators, and status badges.
"""

from typing import Tuple, Optional, Dict, Any, List
import cv2
import numpy as np

import config
from .gesture_detector import GestureResult
from .hand_tracker import HandLandmarksIndex as HLI


class UIRenderer:
    """
    Renders a modern, high-contrast HUD over the live webcam feed.
    """

    def __init__(self) -> None:
        """Initializes UI assets, font choices, and animation states."""
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.font_bold = cv2.FONT_HERSHEY_SIMPLEX
        self.click_effect_frames: int = 0
        self.last_click_pos: Tuple[int, int] = (0, 0)

    def draw_interaction_zone(
        self,
        frame: np.ndarray,
        margin_x: int = config.CAMERA_MARGIN_X,
        margin_y: int = config.CAMERA_MARGIN_Y,
    ) -> None:
        """
        Draws the active interaction zone bounding box with sleek corner brackets.

        Args:
            frame: Video frame (modified in-place).
            margin_x: Horizontal boundary margin in pixels.
            margin_y: Vertical boundary margin in pixels.
        """
        h, w = frame.shape[:2]
        x1 = margin_x
        y1 = margin_y
        x2 = w - margin_x
        y2 = h - margin_y

        # Draw translucent boundary rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), config.COLOR_ZONE, 2)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        # Draw corner accent brackets
        bracket_len = 24
        thick = 3
        c = config.COLOR_ZONE

        # Top-Left corner
        cv2.line(frame, (x1, y1), (x1 + bracket_len, y1), c, thick)
        cv2.line(frame, (x1, y1), (x1, y1 + bracket_len), c, thick)

        # Top-Right corner
        cv2.line(frame, (x2, y1), (x2 - bracket_len, y1), c, thick)
        cv2.line(frame, (x2, y1), (x2, y1 + bracket_len), c, thick)

        # Bottom-Left corner
        cv2.line(frame, (x1, y2), (x1 + bracket_len, y2), c, thick)
        cv2.line(frame, (x1, y2), (x1, y2 - bracket_len), c, thick)

        # Bottom-Right corner
        cv2.line(frame, (x2, y2), (x2 - bracket_len, y2), c, thick)
        cv2.line(frame, (x2, y2), (x2, y2 - bracket_len), c, thick)

        # Corner label
        cv2.putText(
            frame,
            "INTERACTION ZONE",
            (x1 + 8, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            config.COLOR_ZONE,
            1,
            cv2.LINE_AA,
        )

    def draw_hud_panel(
        self,
        frame: np.ndarray,
        gesture_result: GestureResult,
        fps: float,
        is_paused: bool = False,
    ) -> None:
        """
        Renders a clean glassmorphic telemetry card in the top-left corner.

        Args:
            frame: Video frame (modified in-place).
            gesture_result: Telemetry from gesture detection.
            fps: Current measured frames per second.
            is_paused: Whether mouse control is temporarily suspended.
        """
        panel_x = 20
        panel_y = 20
        panel_w = 320
        panel_h = 240

        # Draw translucent dark background card
        sub_img = frame[panel_y : panel_y + panel_h, panel_x : panel_x + panel_w]
        dark_rect = np.full(sub_img.shape, 20, dtype=np.uint8)
        blend = cv2.addWeighted(sub_img, 0.25, dark_rect, 0.75, 0)
        frame[panel_y : panel_y + panel_h, panel_x : panel_x + panel_w] = blend

        # Border around the HUD card
        cv2.rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (70, 70, 85),
            1,
            cv2.LINE_AA,
        )

        # Header Title
        cv2.putText(
            frame,
            "HAND GESTURE MOUSE",
            (panel_x + 14, panel_y + 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            config.COLOR_TEXT_WHITE,
            2,
            cv2.LINE_AA,
        )
        cv2.line(
            frame,
            (panel_x + 14, panel_y + 36),
            (panel_x + panel_w - 14, panel_y + 36),
            (60, 60, 75),
            1,
        )

        # Determine gesture badge style & color
        gesture_name = "PAUSED" if is_paused else gesture_result.gesture
        if gesture_name == "MOVE":
            gesture_color = config.COLOR_MOVE
            cursor_status = "ACTIVE"
            cursor_color = config.COLOR_MOVE
        elif gesture_name == "SCROLL":
            gesture_color = config.COLOR_SCROLL
            cursor_status = "STANDBY"
            cursor_color = config.COLOR_TEXT_DIM
        elif gesture_name == "CLICK":
            gesture_color = config.COLOR_CLICK
            cursor_status = "CLICKING"
            cursor_color = config.COLOR_CLICK
        elif gesture_name == "PAUSED":
            gesture_color = (0, 165, 255)
            cursor_status = "SUSPENDED"
            cursor_color = (0, 165, 255)
        else:
            gesture_color = config.COLOR_IDLE
            cursor_status = "STANDBY"
            cursor_color = config.COLOR_TEXT_DIM

        # Click status string
        if gesture_result.click_event:
            click_status = "TRIGGERED"
            click_color = config.COLOR_CLICK
        elif gesture_result.is_pinched:
            click_status = "PINCHED"
            click_color = (0, 200, 255)
        else:
            click_status = "READY"
            click_color = config.COLOR_TEXT_DIM

        # Scroll status string
        if gesture_result.gesture == "SCROLL":
            if gesture_result.scroll_delta > 0.5:
                scroll_status = "SCROLL UP"
                scroll_color = config.COLOR_SCROLL
            elif gesture_result.scroll_delta < -0.5:
                scroll_status = "SCROLL DOWN"
                scroll_color = config.COLOR_SCROLL
            else:
                scroll_status = "ACTIVE"
                scroll_color = config.COLOR_SCROLL
        else:
            scroll_status = "OFF"
            scroll_color = config.COLOR_TEXT_DIM

        # Render Metrics (Label -> Value)
        entries = [
            ("Gesture:", gesture_name, gesture_color),
            ("Cursor:", cursor_status, cursor_color),
            ("Click:", click_status, click_color),
            ("Scroll:", scroll_status, scroll_color),
            ("FPS:", f"{fps:.1f}", config.COLOR_ACCENT),
        ]

        start_y = panel_y + 65
        row_height = 32

        for i, (label, val, col) in enumerate(entries):
            y_pos = start_y + (i * row_height)
            # Metric label
            cv2.putText(
                frame,
                label,
                (panel_x + 16, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                config.COLOR_TEXT_DIM,
                1,
                cv2.LINE_AA,
            )
            # Metric value
            cv2.putText(
                frame,
                val,
                (panel_x + 120, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                col,
                2,
                cv2.LINE_AA,
            )

    def draw_finger_highlights(
        self,
        frame: np.ndarray,
        landmarks: Optional[List[Tuple[int, int, float]]],
        gesture_result: GestureResult,
    ) -> None:
        """
        Draws dynamic visual cues on the fingertips (halos, pinch lines, and scroll arrows).

        Args:
            frame: Video frame (modified in-place).
            landmarks: 21 pixel-space hand landmarks.
            gesture_result: Telemetry containing pinch and gesture data.
        """
        if landmarks is None or len(landmarks) < 21:
            return

        thumb_tip = (landmarks[HLI.THUMB_TIP][0], landmarks[HLI.THUMB_TIP][1])
        index_tip = (landmarks[HLI.INDEX_TIP][0], landmarks[HLI.INDEX_TIP][1])
        middle_tip = (landmarks[HLI.MIDDLE_TIP][0], landmarks[HLI.MIDDLE_TIP][1])

        # 1. Index Fingertip Cursor Indicator
        if gesture_result.gesture == "MOVE":
            # Outer glowing halo
            cv2.circle(frame, index_tip, 16, config.COLOR_MOVE, 2, cv2.LINE_AA)
            cv2.circle(frame, index_tip, 6, config.COLOR_MOVE, -1, cv2.LINE_AA)
            cv2.putText(
                frame,
                "POINTER",
                (index_tip[0] + 12, index_tip[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                config.COLOR_MOVE,
                1,
                cv2.LINE_AA,
            )
        else:
            cv2.circle(frame, index_tip, 5, (220, 220, 220), -1, cv2.LINE_AA)

        # 2. Pinch Indicator & Measurement Line
        if config.SHOW_PINCH_GAUGE and gesture_result.gesture != "SCROLL":
            # Line connecting thumb and index
            line_color = (
                config.COLOR_CLICK if gesture_result.is_pinched else (120, 180, 255)
            )
            cv2.line(frame, thumb_tip, index_tip, line_color, 2, cv2.LINE_AA)

            # Midpoint marker
            mid_x, mid_y = gesture_result.pinch_midpoint
            cv2.circle(frame, (mid_x, mid_y), 4, line_color, -1, cv2.LINE_AA)

            # Display distance in pixels
            dist_label = f"{int(gesture_result.pinch_distance)}px"
            cv2.putText(
                frame,
                dist_label,
                (mid_x + 10, mid_y + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                line_color,
                1,
                cv2.LINE_AA,
            )

        # 3. Click Visual Ripple Effect
        if gesture_result.click_event:
            self.click_effect_frames = 12
            self.last_click_pos = gesture_result.pinch_midpoint

        if self.click_effect_frames > 0:
            radius = int((13 - self.click_effect_frames) * 3.5)
            cv2.circle(
                frame,
                self.last_click_pos,
                radius,
                config.COLOR_CLICK,
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "LEFT CLICK",
                (self.last_click_pos[0] - 40, self.last_click_pos[1] - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                config.COLOR_CLICK,
                2,
                cv2.LINE_AA,
            )
            self.click_effect_frames -= 1

        # 4. Scroll Visual Feedback
        if gesture_result.gesture == "SCROLL":
            # Draw neutral anchor line across interaction zone if active
            if gesture_result.scroll_anchor_y is not None:
                anchor_y = int(gesture_result.scroll_anchor_y)
                h, w = frame.shape[:2]
                x1 = config.CAMERA_MARGIN_X
                x2 = w - config.CAMERA_MARGIN_X

                # Cyan neutral line across interaction zone (matching UI reference)
                cv2.line(frame, (x1, anchor_y), (x2, anchor_y), (255, 235, 50), 2, cv2.LINE_AA)
                cv2.putText(
                    frame,
                    "- NEUTRAL",
                    (x1 + (x2 - x1) // 2 - 45, anchor_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (255, 235, 50),
                    2,
                    cv2.LINE_AA,
                )

            # Connect index and middle tips
            cv2.line(frame, index_tip, middle_tip, config.COLOR_SCROLL, 3, cv2.LINE_AA)
            cv2.circle(frame, index_tip, 9, config.COLOR_SCROLL, -1, cv2.LINE_AA)
            cv2.circle(frame, middle_tip, 9, config.COLOR_SCROLL, -1, cv2.LINE_AA)

            # Scroll direction label
            scroll_mid_y = int((index_tip[1] + middle_tip[1]) / 2)
            scroll_mid_x = int((index_tip[0] + middle_tip[0]) / 2)

            # Draw vertical guide line between finger position and neutral line
            if gesture_result.scroll_anchor_y is not None:
                cv2.line(
                    frame,
                    (scroll_mid_x, scroll_mid_y),
                    (scroll_mid_x, int(gesture_result.scroll_anchor_y)),
                    (0, 215, 255),
                    1,
                    cv2.LINE_AA,
                )

            if gesture_result.scroll_delta > 0.5:
                direction_text = f"▲ SCROLLING UP ({abs(gesture_result.scroll_delta):.1f}x)"
            elif gesture_result.scroll_delta < -0.5:
                direction_text = f"▼ SCROLLING DOWN ({abs(gesture_result.scroll_delta):.1f}x)"
            else:
                direction_text = "— NEUTRAL (HOLD TO PAUSE)"

            cv2.putText(
                frame,
                direction_text,
                (scroll_mid_x - 90, scroll_mid_y - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                config.COLOR_SCROLL,
                2,
                cv2.LINE_AA,
            )

    def draw_footer_bar(
        self,
        frame: np.ndarray,
        is_recording: bool = False,
        record_duration: float = 0.0,
        snapshot_flash: bool = False,
    ) -> None:
        """Draws shortcut guide, snapshot notification, and live recording badge."""
        h, w = frame.shape[:2]
        cv2.putText(
            frame,
            "Press [Q] Quit  |  [P] Pause  |  [R] Record  |  [S] Snapshot",
            (24, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (170, 170, 180),
            1,
            cv2.LINE_AA,
        )

        # Display snapshot feedback banner
        if snapshot_flash:
            snap_text = "[SNAPSHOT SAVED]"
            cv2.putText(
                frame,
                snap_text,
                (w // 2 - 80, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 255, 120),
                2,
                cv2.LINE_AA,
            )

        # Draw live recording badge at top-left
        if is_recording:
            pulse = int(record_duration * 2) % 2 == 0
            rec_color = config.COLOR_REC if pulse else (80, 80, 200)
            cv2.circle(frame, (32, 32), 7, rec_color, -1, cv2.LINE_AA)
            cv2.circle(frame, (32, 32), 9, (255, 255, 255), 1, cv2.LINE_AA)

            mins = int(record_duration) // 60
            secs = int(record_duration) % 60
            rec_text = f"REC {mins:02d}:{secs:02d}"
            cv2.putText(
                frame,
                rec_text,
                (48, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                config.COLOR_REC,
                2,
                cv2.LINE_AA,
            )

