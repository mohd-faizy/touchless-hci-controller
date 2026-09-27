"""
Unit test suite for the Hand Gesture Mouse Controller.
Tests finger state detection, gesture classification priority, debounced click
state machines, coordinate mapping, exponential smoothing, and scroll calculation.
"""

import time
import unittest
from typing import List, Tuple
import numpy as np

from src.gesture_detector import GestureDetector, GestureResult
from src.mouse_controller import MouseController
from src.hand_tracker import HandLandmarksIndex as HLI


def create_synthetic_hand(
    thumb_extended: bool = False,
    index_extended: bool = False,
    middle_extended: bool = False,
    ring_extended: bool = False,
    pinky_extended: bool = False,
    pinch: bool = False,
) -> List[Tuple[int, int, float]]:
    """
    Generates a realistic 21-landmark synthetic hand configuration in pixel space.
    Wrist is positioned at (300, 450).
    """
    # 21 landmarks initialized to default neutral hand
    landmarks: List[Tuple[int, int, float]] = [(0, 0, 0.0)] * 21

    # Wrist at base of hand
    landmarks[HLI.WRIST] = (300, 450, 0.0)

    # Base MCP joints (knuckles)
    landmarks[HLI.THUMB_CMC] = (250, 420, 0.0)
    landmarks[HLI.THUMB_MCP] = (230, 390, 0.0)
    landmarks[HLI.INDEX_MCP] = (270, 340, 0.0)
    landmarks[HLI.MIDDLE_MCP] = (300, 335, 0.0)
    landmarks[HLI.RING_MCP] = (330, 340, 0.0)
    landmarks[HLI.PINKY_MCP] = (355, 355, 0.0)

    # Index finger
    landmarks[HLI.INDEX_PIP] = (270, 290, 0.0)
    landmarks[HLI.INDEX_DIP] = (270, 250, 0.0)
    if index_extended:
        landmarks[HLI.INDEX_TIP] = (270, 200, 0.0)  # Far from wrist & MCP
    else:
        landmarks[HLI.INDEX_TIP] = (270, 320, 0.0)  # Curled towards palm

    # Middle finger
    landmarks[HLI.MIDDLE_PIP] = (300, 285, 0.0)
    landmarks[HLI.MIDDLE_DIP] = (300, 245, 0.0)
    if middle_extended:
        landmarks[HLI.MIDDLE_TIP] = (300, 195, 0.0)
    else:
        landmarks[HLI.MIDDLE_TIP] = (300, 320, 0.0)

    # Ring finger
    landmarks[HLI.RING_PIP] = (330, 290, 0.0)
    landmarks[HLI.RING_DIP] = (330, 255, 0.0)
    if ring_extended:
        landmarks[HLI.RING_TIP] = (330, 210, 0.0)
    else:
        landmarks[HLI.RING_TIP] = (330, 325, 0.0)

    # Pinky finger
    landmarks[HLI.PINKY_PIP] = (355, 305, 0.0)
    landmarks[HLI.PINKY_DIP] = (355, 275, 0.0)
    if pinky_extended:
        landmarks[HLI.PINKY_TIP] = (355, 235, 0.0)
    else:
        landmarks[HLI.PINKY_TIP] = (355, 335, 0.0)

    # Thumb
    landmarks[HLI.THUMB_IP] = (210, 360, 0.0)
    if pinch:
        # Position thumb tip very close to index tip
        idx_tip = landmarks[HLI.INDEX_TIP]
        landmarks[HLI.THUMB_TIP] = (idx_tip[0] - 15, idx_tip[1] + 10, 0.0)
    elif thumb_extended:
        landmarks[HLI.THUMB_TIP] = (175, 330, 0.0)  # Reaches out sideways
    else:
        landmarks[HLI.THUMB_TIP] = (240, 380, 0.0)  # Tucked against index MCP

    return landmarks


class TestGestureDetector(unittest.TestCase):
    """Test suite for gesture detection rules, state transitions, and debouncing."""

    def setUp(self) -> None:
        self.detector = GestureDetector(
            pinch_threshold=38.0,
            pinch_release_threshold=48.0,
            click_cooldown=0.30,
            scroll_sensitivity=2.0,
            scroll_deadzone=4.0,
        )

    def test_finger_states_index_only(self) -> None:
        """Verifies only index finger is detected as extended."""
        hand = create_synthetic_hand(index_extended=True)
        states = self.detector.get_finger_states(hand)
        self.assertTrue(states["index"])
        self.assertFalse(states["middle"])
        self.assertFalse(states["ring"])
        self.assertFalse(states["pinky"])

    def test_move_gesture_classification(self) -> None:
        """Verifies MOVE gesture is classified when index is extended alone."""
        hand = create_synthetic_hand(index_extended=True)
        result = self.detector.update(hand)
        self.assertEqual(result.gesture, "MOVE")
        self.assertIsNotNone(result.cursor_pos)
        self.assertFalse(result.click_event)
        self.assertEqual(result.scroll_delta, 0.0)

    def test_scroll_mode_classification(self) -> None:
        """Verifies SCROLL mode when both index and middle fingers are extended."""
        hand = create_synthetic_hand(index_extended=True, middle_extended=True)
        result = self.detector.update(hand)
        self.assertEqual(result.gesture, "SCROLL")
        self.assertFalse(result.click_event)

    def test_scroll_movement_calculation(self) -> None:
        """Verifies vertical displacement translates to correct scroll delta."""
        # Initial frame in scroll mode
        hand_frame1 = create_synthetic_hand(index_extended=True, middle_extended=True)
        self.detector.update(hand_frame1)

        # Move hand UP in frame 2 (Y decreases from ~200 to ~180: delta = -20px)
        hand_frame2 = []
        for (x, y, z) in hand_frame1:
            hand_frame2.append((x, y - 20, z))

        result2 = self.detector.update(hand_frame2)
        self.assertEqual(result2.gesture, "SCROLL")
        # Inverted: Hand moving UP should yield positive scroll amount
        self.assertGreater(result2.scroll_delta, 0.0)

    def test_scroll_deadzone(self) -> None:
        """Verifies micro-displacements under scroll deadzone are ignored."""
        hand1 = create_synthetic_hand(index_extended=True, middle_extended=True)
        self.detector.update(hand1)

        # Very tiny 1px movement
        hand2 = [(x, y - 1, z) for (x, y, z) in hand1]
        result2 = self.detector.update(hand2)
        self.assertEqual(result2.scroll_delta, 0.0)

    def test_pinch_click_debouncing_and_cooldown(self) -> None:
        """Verifies click state machine: PINCH START triggers once, PINCH HELD does not repeat."""
        pinch_hand = create_synthetic_hand(index_extended=True, pinch=True)

        # Frame 1: Pinch Start -> Click event must trigger
        res1 = self.detector.update(pinch_hand)
        self.assertEqual(res1.gesture, "CLICK")
        self.assertTrue(res1.click_event, "Click event should fire on pinch start")

        # Frame 2: Pinch Held -> Click event must NOT trigger again
        res2 = self.detector.update(pinch_hand)
        self.assertEqual(res2.gesture, "CLICK")
        self.assertFalse(res2.click_event, "Click event must not repeat while held")

        # Frame 3: Pinch Released (open hand)
        open_hand = create_synthetic_hand(index_extended=True, pinch=False)
        res3 = self.detector.update(open_hand)
        self.assertFalse(res3.is_pinched)
        self.assertFalse(res3.click_event)

        # Frame 4: Pinch again immediately within cooldown -> rejected
        res4 = self.detector.update(pinch_hand)
        self.assertFalse(res4.click_event, "Immediate re-pinch within cooldown must be debounced")

        # Frame 5: Pinch after cooldown has elapsed -> accepted
        time.sleep(0.35)
        self.detector.update(open_hand)  # ensure release
        res5 = self.detector.update(pinch_hand)
        self.assertTrue(res5.click_event, "Pinch after cooldown should fire a new click event")

    def test_gesture_priority_scroll_over_click(self) -> None:
        """Verifies that SCROLL mode strictly overrides pinch/click actions."""
        # Index + Middle extended with tips near thumb
        hand = create_synthetic_hand(index_extended=True, middle_extended=True, pinch=True)
        result = self.detector.update(hand)
        self.assertEqual(result.gesture, "SCROLL")
        self.assertFalse(result.click_event)


class TestMouseController(unittest.TestCase):
    """Test suite for coordinate transformation, clamping, and exponential smoothing."""

    def setUp(self) -> None:
        self.mouse = MouseController(
            smoothing_alpha=0.25,
            margin_x=100,
            margin_y=100,
        )

    def test_coordinate_mapping(self) -> None:
        """Verifies mapping from camera rectangle to full screen dimensions."""
        cam_w, cam_h = 1280, 720
        scr_w, scr_h = self.mouse.screen_width, self.mouse.screen_height

        # Point exactly at top-left margin -> should map to (0, 0)
        sx, sy = self.mouse.map_coordinates(100, 100, cam_w, cam_h)
        self.assertAlmostEqual(sx, 0.0, places=1)
        self.assertAlmostEqual(sy, 0.0, places=1)

        # Point exactly at bottom-right margin -> should map to screen_w-1, screen_h-1
        sx, sy = self.mouse.map_coordinates(cam_w - 100, cam_h - 100, cam_w, cam_h)
        self.assertAlmostEqual(sx, scr_w - 1, places=1)
        self.assertAlmostEqual(sy, scr_h - 1, places=1)

        # Point outside bounds -> clamped correctly
        sx, sy = self.mouse.map_coordinates(0, 0, cam_w, cam_h)
        self.assertEqual(sx, 0.0)
        self.assertEqual(sy, 0.0)

    def test_exponential_smoothing(self) -> None:
        """Verifies EMA low-pass filtering math."""
        # First call initializes without jumping from 0
        s1_x, s1_y = self.mouse.smooth_coordinates(100.0, 200.0)
        self.assertEqual(s1_x, 100.0)
        self.assertEqual(s1_y, 200.0)

        # Second call: smoothed = 0.25 * target + 0.75 * prev
        # target = (200, 300) -> smoothed = 0.25 * 200 + 0.75 * 100 = 50 + 75 = 125
        s2_x, s2_y = self.mouse.smooth_coordinates(200.0, 300.0)
        self.assertAlmostEqual(s2_x, 125.0, places=2)
        self.assertAlmostEqual(s2_y, 225.0, places=2)

    def test_adaptive_smoothing(self) -> None:
        """Verifies adaptive smoothing scales alpha between alpha_min and alpha_max."""
        # Initialize
        self.mouse.reset()
        self.mouse.smooth_coordinates(100.0, 100.0)

        # Micro-movement (< 5px) should use alpha_min (0.12)
        target_small = (103.0, 103.0)
        s_small_x, s_small_y = self.mouse.smooth_coordinates(
            target_small[0], target_small[1], use_adaptive=True
        )
        expected_small_x = 0.12 * 103.0 + 0.88 * 100.0
        self.assertAlmostEqual(s_small_x, expected_small_x, places=1)

        # Large sweep (> 65px) should use alpha_max (0.55)
        self.mouse.reset()
        self.mouse.smooth_coordinates(100.0, 100.0)
        target_large = (300.0, 300.0)
        s_large_x, s_large_y = self.mouse.smooth_coordinates(
            target_large[0], target_large[1], use_adaptive=True
        )
        expected_large_x = 0.55 * 300.0 + 0.45 * 100.0
        self.assertAlmostEqual(s_large_x, expected_large_x, places=1)


if __name__ == "__main__":
    unittest.main()
