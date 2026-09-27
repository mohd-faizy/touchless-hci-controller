import os
import warnings

# Suppress noisy external third-party C++ logs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")

# Suppress deprecation warning from protobuf symbol_database and route to modern message_factory
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")
try:
    from google.protobuf import symbol_database, message_factory
    if hasattr(symbol_database.SymbolDatabase, "GetPrototype"):
        symbol_database.SymbolDatabase.GetPrototype = lambda self, descriptor: message_factory.GetMessageClass(descriptor)
except Exception:
    pass

from typing import List, Tuple, Optional, Dict, Any
import cv2
import numpy as np
import mediapipe as mp


class HandLandmarksIndex:
    """Standard MediaPipe Hand 21 Landmark Indices."""
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


class HandTracker:
    """
    Manages MediaPipe Hands initialization, landmark extraction, and visual drawing.
    """

    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.70,
        min_tracking_confidence: float = 0.70,
    ) -> None:
        """
        Initializes the MediaPipe Hands detector.

        Args:
            max_num_hands: Maximum number of hands to track concurrently.
            min_detection_confidence: Confidence threshold for initial hand detection.
            min_tracking_confidence: Confidence threshold for subsequent landmark tracking.
        """
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # Initialize MediaPipe solutions
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.max_num_hands,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Custom drawing styles for clean futuristic HUD aesthetic
        self.landmark_drawing_spec = self.mp_drawing.DrawingSpec(
            color=(0, 255, 200), thickness=2, circle_radius=3
        )
        self.connection_drawing_spec = self.mp_drawing.DrawingSpec(
            color=(240, 240, 240), thickness=2, circle_radius=1
        )

        self.last_results: Optional[Any] = None

    def process_frame(self, frame_bgr: np.ndarray) -> Optional[Any]:
        """
        Processes a single BGR OpenCV frame and detects hand landmarks.

        Args:
            frame_bgr: Raw input frame from camera in BGR format.

        Returns:
            MediaPipe process results object, or None if frame is invalid.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return None

        # MediaPipe requires RGB format
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Optimize performance: mark image as non-writeable to pass by reference
        frame_rgb.flags.writeable = False
        self.last_results = self.hands.process(frame_rgb)
        frame_rgb.flags.writeable = True

        return self.last_results

    def get_hand_landmarks(
        self,
        frame_shape: Tuple[int, int, int],
        hand_index: int = 0
    ) -> Optional[List[Tuple[int, int, float]]]:
        """
        Extracts pixel coordinates and depth for landmarks of a detected hand.

        Args:
            frame_shape: Shape tuple of the frame (height, width, channels).
            hand_index: Index of the detected hand (0 for primary hand).

        Returns:
            List of 21 tuples [(x_px, y_px, z_normalized), ...] or None if no hand detected.
        """
        if not self.last_results or not self.last_results.multi_hand_landmarks:
            return None

        if hand_index >= len(self.last_results.multi_hand_landmarks):
            return None

        hand_landmarks = self.last_results.multi_hand_landmarks[hand_index]
        h, w = frame_shape[0], frame_shape[1]

        landmarks_px: List[Tuple[int, int, float]] = []
        for lm in hand_landmarks.landmark:
            # Convert normalized [0.0, 1.0] coordinates to integer pixel space
            px_x = int(lm.x * w)
            px_y = int(lm.y * h)
            landmarks_px.append((px_x, px_y, lm.z))

        return landmarks_px

    def get_handedness(self, hand_index: int = 0) -> str:
        """
        Returns the classified handedness ("Right" or "Left") of the detected hand.
        """
        if (
            self.last_results
            and self.last_results.multi_handedness
            and hand_index < len(self.last_results.multi_handedness)
        ):
            return self.last_results.multi_handedness[hand_index].classification[0].label
        return "Right"

    def draw_landmarks(
        self,
        frame_bgr: np.ndarray,
        hand_index: int = 0,
        draw_connections: bool = True
    ) -> None:
        """
        Draws hand landmarks and connections directly onto the provided frame.

        Args:
            frame_bgr: Target OpenCV image buffer (modified in-place).
            hand_index: Hand index to render.
            draw_connections: Whether to draw connecting skeleton lines.
        """
        if (
            self.last_results
            and self.last_results.multi_hand_landmarks
            and hand_index < len(self.last_results.multi_hand_landmarks)
        ):
            hand_landmarks = self.last_results.multi_hand_landmarks[hand_index]
            connections = self.mp_hands.HAND_CONNECTIONS if draw_connections else None

            self.mp_drawing.draw_landmarks(
                image=frame_bgr,
                landmark_list=hand_landmarks,
                connections=connections,
                landmark_drawing_spec=self.landmark_drawing_spec,
                connection_drawing_spec=self.connection_drawing_spec,
            )

    def close(self) -> None:
        """Releases MediaPipe resources."""
        if hasattr(self, 'hands') and self.hands:
            self.hands.close()
