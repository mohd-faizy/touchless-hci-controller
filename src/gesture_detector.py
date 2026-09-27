"""
Gesture detection module for the Invisible Hand Gesture Mouse Controller.
Implements robust multi-cue finger state detection, hysteresis pinch sensing,
gesture priority conflict resolution, and debounced state machines.
"""

import time
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
import numpy as np

from .hand_tracker import HandLandmarksIndex as HLI


@dataclass
class GestureResult:
    """Encapsulates the complete gesture detection state for a single video frame."""
    gesture: str = "IDLE"  # "IDLE", "MOVE", "CLICK", "SCROLL"
    cursor_pos: Optional[Tuple[int, int]] = None
    pinch_distance: float = 999.0
    is_pinched: bool = False
    finger_states: Dict[str, bool] = field(
        default_factory=lambda: {
            "thumb": False,
            "index": False,
            "middle": False,
            "ring": False,
            "pinky": False,
        }
    )
    scroll_delta: float = 0.0
    click_event: bool = False
    pinch_midpoint: Tuple[int, int] = (0, 0)
    confidence: float = 1.0
    scroll_anchor_y: Optional[float] = None


class GestureDetector:
    """
    Evaluates hand landmarks to classify gestures with priority resolution
    and robust state machines.
    """

    def __init__(
        self,
        pinch_threshold: float = 38.0,
        pinch_release_threshold: float = 48.0,
        click_cooldown: float = 0.40,
        scroll_sensitivity: float = 2.4,
        scroll_deadzone: float = 4.0,
        scroll_speed_cap: int = 25,
    ) -> None:
        """
        Initializes gesture detector parameters and internal state variables.

        Args:
            pinch_threshold: Euclidean pixel distance to trigger a pinch.
            pinch_release_threshold: Distance to release pinch (hysteresis).
            click_cooldown: Minimum seconds between successive click events.
            scroll_sensitivity: Multiplier for scroll delta calculation.
            scroll_deadzone: Minimum pixel movement to register scrolling.
            scroll_speed_cap: Maximum scroll clicks permitted per frame.
        """
        self.pinch_threshold = pinch_threshold
        self.pinch_release_threshold = pinch_release_threshold
        self.click_cooldown = click_cooldown
        self.scroll_sensitivity = scroll_sensitivity
        self.scroll_deadzone = scroll_deadzone
        self.scroll_speed_cap = scroll_speed_cap

        # State tracking variables
        self.current_gesture: str = "IDLE"
        self.previous_gesture: str = "IDLE"

        # Pinch & Click state machine
        self.pinch_active: bool = False
        self.last_click_time: float = 0.0
        self.click_latched: bool = False

        # Scroll state tracking
        self.scroll_anchor_y: Optional[float] = None
        self.prev_scroll_y: Optional[float] = None
        self.smoothed_scroll_delta: float = 0.0

        # Debouncing and persistence tracking
        self.idle_frame_count: int = 0
        self.last_valid_cursor_pos: Optional[Tuple[int, int]] = None

    @staticmethod
    def _calc_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Computes 2D Euclidean distance between two points."""
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def get_finger_states(
        self,
        landmarks: List[Tuple[int, int, float]],
        handedness: str = "Right"
    ) -> Dict[str, bool]:
        """
        Determines whether each finger is extended or curled.
        
        Uses anatomical landmark relationships (distance from wrist and MCP/PIP joints)
        rather than simple Y-coordinates, ensuring stable detection even when the
        hand is tilted or at different distances from the camera.

        Args:
            landmarks: 21 pixel-space hand landmarks [(x, y, z), ...].
            handedness: "Right" or "Left" hand label.

        Returns:
            Dictionary mapping finger names to boolean extended status.
        """
        if len(landmarks) < 21:
            return {"thumb": False, "index": False, "middle": False, "ring": False, "pinky": False}

        wrist = (landmarks[HLI.WRIST][0], landmarks[HLI.WRIST][1])
        pinky_mcp = (landmarks[HLI.PINKY_MCP][0], landmarks[HLI.PINKY_MCP][1])

        # Finger definitions: (Tip, DIP, PIP, MCP)
        fingers = {
            "index": (HLI.INDEX_TIP, HLI.INDEX_DIP, HLI.INDEX_PIP, HLI.INDEX_MCP),
            "middle": (HLI.MIDDLE_TIP, HLI.MIDDLE_DIP, HLI.MIDDLE_PIP, HLI.MIDDLE_MCP),
            "ring": (HLI.RING_TIP, HLI.RING_DIP, HLI.RING_PIP, HLI.RING_MCP),
            "pinky": (HLI.PINKY_TIP, HLI.PINKY_DIP, HLI.PINKY_PIP, HLI.PINKY_MCP),
        }

        states: Dict[str, bool] = {}

        # 1. Four main fingers (Index, Middle, Ring, Pinky)
        # An extended finger's tip is significantly further from the wrist and MCP than PIP
        for finger_name, (tip_idx, dip_idx, pip_idx, mcp_idx) in fingers.items():
            tip = (landmarks[tip_idx][0], landmarks[tip_idx][1])
            dip = (landmarks[dip_idx][0], landmarks[dip_idx][1])
            pip = (landmarks[pip_idx][0], landmarks[pip_idx][1])
            mcp = (landmarks[mcp_idx][0], landmarks[mcp_idx][1])

            dist_tip_wrist = self._calc_distance(tip, wrist)
            dist_pip_wrist = self._calc_distance(pip, wrist)
            dist_tip_mcp = self._calc_distance(tip, mcp)
            dist_pip_mcp = self._calc_distance(pip, mcp)

            # Finger is extended if tip is further from wrist and MCP than PIP
            is_extended_distance = (dist_tip_wrist > dist_pip_wrist * 1.02) and (dist_tip_mcp > dist_pip_mcp * 1.02)

            # Secondary directional check: tip should be further outward than pip along the mcp->pip axis
            vec_mcp_pip = np.array([pip[0] - mcp[0], pip[1] - mcp[1]], dtype=float)
            vec_pip_tip = np.array([tip[0] - pip[0], tip[1] - pip[1]], dtype=float)
            dot_alignment = np.dot(vec_mcp_pip, vec_pip_tip)

            states[finger_name] = bool(is_extended_distance and dot_alignment > 0)

        # 2. Thumb extension
        # Thumb articulates outward horizontally from the palm.
        # Check distance between thumb tip and pinky base vs thumb IP and pinky base
        thumb_tip = (landmarks[HLI.THUMB_TIP][0], landmarks[HLI.THUMB_TIP][1])
        thumb_ip = (landmarks[HLI.THUMB_IP][0], landmarks[HLI.THUMB_IP][1])
        thumb_mcp = (landmarks[HLI.THUMB_MCP][0], landmarks[HLI.THUMB_MCP][1])

        dist_thumb_tip_pinky = self._calc_distance(thumb_tip, pinky_mcp)
        dist_thumb_ip_pinky = self._calc_distance(thumb_ip, pinky_mcp)
        dist_thumb_tip_mcp = self._calc_distance(thumb_tip, thumb_mcp)
        dist_thumb_ip_mcp = self._calc_distance(thumb_ip, thumb_mcp)

        # Extended thumb reaches outward away from palm and MCP
        thumb_extended = (dist_thumb_tip_pinky > dist_thumb_ip_pinky * 1.08) and (
            dist_thumb_tip_mcp > dist_thumb_ip_mcp * 1.05
        )
        states["thumb"] = bool(thumb_extended)

        return states

    def update(
        self,
        landmarks: Optional[List[Tuple[int, int, float]]],
        handedness: str = "Right"
    ) -> GestureResult:
        """
        Main gesture evaluation pipeline for a frame.

        Applies strict priority order:
        1. SCROLL (Index + Middle extended, others curled)
        2. PINCH CLICK (Thumb and Index tip close together, others curled)
        3. CURSOR MOVE (Only Index extended)
        4. IDLE (Fist, open hand, or other poses)

        Args:
            landmarks: 21 landmark tuples or None if no hand detected.
            handedness: Hand classification ("Right" or "Left").

        Returns:
            GestureResult with classified mode and action payload.
        """
        # When no hand is in view, transition cleanly to IDLE
        if landmarks is None or len(landmarks) < 21:
            self.current_gesture = "IDLE"
            self.previous_gesture = "IDLE"
            self.prev_scroll_y = None
            self.pinch_active = False
            self.click_latched = False
            return GestureResult(gesture="IDLE")

        now = time.time()
        finger_states = self.get_finger_states(landmarks, handedness)

        thumb_tip = (landmarks[HLI.THUMB_TIP][0], landmarks[HLI.THUMB_TIP][1])
        index_tip = (landmarks[HLI.INDEX_TIP][0], landmarks[HLI.INDEX_TIP][1])
        middle_tip = (landmarks[HLI.MIDDLE_TIP][0], landmarks[HLI.MIDDLE_TIP][1])

        # Calculate Euclidean pinch distance and midpoint
        pinch_dist = self._calc_distance(thumb_tip, index_tip)
        pinch_midpoint = (
            int((thumb_tip[0] + index_tip[0]) / 2),
            int((thumb_tip[1] + index_tip[1]) / 2),
        )

        # Hysteresis for pinch detection
        if not self.pinch_active:
            if pinch_dist < self.pinch_threshold:
                self.pinch_active = True
        else:
            if pinch_dist > self.pinch_release_threshold:
                self.pinch_active = False
                self.click_latched = False  # Reset latch on release

        idx_extended = finger_states["index"]
        mid_extended = finger_states["middle"]
        ring_extended = finger_states["ring"]
        pnk_extended = finger_states["pinky"]

        result = GestureResult(
            finger_states=finger_states,
            pinch_distance=pinch_dist,
            is_pinched=self.pinch_active,
            pinch_midpoint=pinch_midpoint,
        )

        # ----------------------------------------------------------------------
        # PRIORITY 1: SCROLL MODE
        # Both Index and Middle fingers are raised while Ring & Pinky are down
        # ----------------------------------------------------------------------
        if idx_extended and mid_extended and not ring_extended and not pnk_extended:
            self.current_gesture = "SCROLL"
            result.gesture = "SCROLL"

            # In scroll mode, disallow pinch clicks and normal cursor movement
            self.click_latched = False

            # Calculate vertical hand position (average of index and middle tips)
            current_y = (index_tip[1] + middle_tip[1]) / 2.0

            # On first frame entering SCROLL mode, establish neutral anchor line
            if self.scroll_anchor_y is None:
                self.scroll_anchor_y = current_y

            result.scroll_anchor_y = self.scroll_anchor_y

            # Calculate displacement relative to neutral line (upward is positive displacement)
            displacement = self.scroll_anchor_y - current_y

            # Apply neutral deadzone
            if abs(displacement) > self.scroll_deadzone:
                excess = displacement - np.sign(displacement) * self.scroll_deadzone
                # Progressive rate: gentle near neutral line, accelerating smoothly when held further
                speed = np.sign(displacement) * (1.0 + abs(excess / 10.0) ** 1.15) * self.scroll_sensitivity
                scroll_value = float(np.clip(speed, -self.scroll_speed_cap, self.scroll_speed_cap))
                result.scroll_delta = scroll_value
            else:
                result.scroll_delta = 0.0

            self.prev_scroll_y = current_y
            self.previous_gesture = "SCROLL"
            return result

        # Reset scroll tracking & neutral anchor when exiting scroll mode
        self.scroll_anchor_y = None
        self.prev_scroll_y = None

        # ----------------------------------------------------------------------
        # PRIORITY 2: PINCH CLICK
        # Thumb and Index tips are pinched together, while Ring and Pinky are down
        # ----------------------------------------------------------------------
        if self.pinch_active and not ring_extended and not pnk_extended:
            self.current_gesture = "CLICK"
            result.gesture = "CLICK"
            result.cursor_pos = index_tip

            # Click state machine with debouncing
            # Trigger once on pinch onset; hold does nothing; release resets
            if not self.click_latched and (now - self.last_click_time >= self.click_cooldown):
                result.click_event = True
                self.click_latched = True
                self.last_click_time = now
            else:
                result.click_event = False

            self.previous_gesture = "CLICK"
            return result

        # If pinch is released, unlatch click
        if not self.pinch_active:
            self.click_latched = False

        # ----------------------------------------------------------------------
        # PRIORITY 3: CURSOR MOVEMENT
        # Only the Index finger is raised; Middle, Ring, Pinky curled; not pinched
        # ----------------------------------------------------------------------
        is_pointing_pose = idx_extended and not mid_extended and not ring_extended and not pnk_extended and not self.pinch_active

        if is_pointing_pose:
            self.current_gesture = "MOVE"
            result.gesture = "MOVE"
            result.cursor_pos = index_tip
            self.previous_gesture = "MOVE"
            self.idle_frame_count = 0
            self.last_valid_cursor_pos = index_tip
            return result

        # Temporal gesture debouncing:
        # If we were just in MOVE mode and gesture momentarily dropped (1-2 noisy frames)
        # without being a pinch or scroll, preserve MOVE mode to prevent cursor stuttering.
        if (
            self.previous_gesture == "MOVE"
            and not self.pinch_active
            and self.idle_frame_count < 2
        ):
            self.idle_frame_count += 1
            self.current_gesture = "MOVE"
            result.gesture = "MOVE"
            result.cursor_pos = index_tip
            return result

        # ----------------------------------------------------------------------
        # PRIORITY 4: IDLE
        # Open hand, fist, or any unmapped pose
        # ----------------------------------------------------------------------
        self.idle_frame_count = 0
        self.current_gesture = "IDLE"
        result.gesture = "IDLE"
        self.previous_gesture = "IDLE"
        return result

    def reset(self) -> None:
        """Resets all internal state and timers."""
        self.current_gesture = "IDLE"
        self.previous_gesture = "IDLE"
        self.pinch_active = False
        self.click_latched = False
        self.prev_scroll_y = None
        self.scroll_anchor_y = None
        self.last_click_time = 0.0
        self.idle_frame_count = 0
