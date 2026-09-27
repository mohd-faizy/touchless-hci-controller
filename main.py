"""
Main application entry point for the Invisible Hand Gesture Mouse Controller.
Integrates webcam video stream, MediaPipe hand tracking, gesture classification,
exponentially smoothed PyAutoGUI mouse control, and real-time HUD rendering.
"""

import os
import sys
import time
import warnings
import argparse
from typing import Optional

# Suppress external third-party C++ logs & protobuf deprecation warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")

import cv2

import config
from src.hand_tracker import HandTracker
from src.gesture_detector import GestureDetector, GestureResult
from src.mouse_controller import MouseController
from src.ui import UIRenderer


def parse_arguments() -> argparse.Namespace:
    """Parses optional command-line overrides for runtime configuration."""
    parser = argparse.ArgumentParser(
        description="Invisible Hand Gesture Mouse Controller"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=config.CAMERA_INDEX,
        help=f"Webcam index (default: {config.CAMERA_INDEX})",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=config.SMOOTHING_ALPHA,
        help=f"Smoothing factor alpha (0.10 to 0.50, default: {config.SMOOTHING_ALPHA})",
    )
    parser.add_argument(
        "--pinch-threshold",
        type=float,
        default=config.PINCH_THRESHOLD,
        help=f"Pinch activation distance in px (default: {config.PINCH_THRESHOLD})",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Disable camera mirroring",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Start recording webcam feed to MP4 video immediately upon launch",
    )
    parser.add_argument(
        "--output-video",
        type=str,
        default=config.DEFAULT_VIDEO_PATH,
        help=f"Output video file path (default: {config.DEFAULT_VIDEO_PATH})",
    )
    return parser.parse_args()


def init_camera(camera_index: int, width: int, height: int) -> cv2.VideoCapture:
    """
    Initializes and configures the webcam capture object with fallback diagnostics.

    Args:
        camera_index: Index of the camera device.
        width: Desired video frame width.
        height: Desired video frame height.

    Returns:
        Configured cv2.VideoCapture object.

    Raises:
        RuntimeError: If webcam cannot be opened.
    """
    print(f"[INFO] Initializing webcam device (index {camera_index})...")
    
    # On Windows, try cv2.CAP_DSHOW for faster startup, fallback to default
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(
            f"Error: Unable to open webcam at index {camera_index}.\n"
            "Troubleshooting steps:\n"
            "  1. Ensure your webcam is securely connected and not in use by another app.\n"
            "  2. Try passing --camera 1 or --camera 2 if using an external USB camera.\n"
            "  3. Check Windows Camera Privacy Settings to ensure apps have camera access."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, config.TARGET_FPS)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Webcam successfully opened with resolution: {actual_w}x{actual_h}")
    return cap


def main() -> None:
    """Core execution loop for the gesture controller."""
    args = parse_arguments()

    print("=" * 65)
    print(" INVISIBLE HAND GESTURE MOUSE CONTROLLER")
    print("=" * 65)
    print(f" Camera Index       : {args.camera}")
    print(f" Smoothing Alpha    : {args.smoothing}")
    print(f" Pinch Threshold    : {args.pinch_threshold} px")
    print(f" Controls           : [Q] Quit  |  [P] Pause  |  [R] Record  |  [S] Snapshot")
    print("=" * 65)

    # Initialize sub-systems
    try:
        cap = init_camera(args.camera, config.FRAME_WIDTH, config.FRAME_HEIGHT)
    except RuntimeError as err:
        print(f"\n[FATAL] {err}\n", file=sys.stderr)
        sys.exit(1)

    tracker = HandTracker(
        max_num_hands=config.MAX_NUM_HANDS,
        min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
    )

    detector = GestureDetector(
        pinch_threshold=args.pinch_threshold,
        pinch_release_threshold=config.PINCH_RELEASE_THRESHOLD,
        click_cooldown=config.CLICK_COOLDOWN,
        scroll_sensitivity=config.SCROLL_SENSITIVITY,
        scroll_deadzone=config.SCROLL_DEADZONE,
        scroll_speed_cap=config.SCROLL_SPEED_CAP,
    )

    mouse = MouseController(
        smoothing_alpha=args.smoothing,
        margin_x=config.CAMERA_MARGIN_X,
        margin_y=config.CAMERA_MARGIN_Y,
        min_cursor_delta=config.MIN_CURSOR_DELTA,
        alpha_min=config.SMOOTHING_ALPHA_MIN,
        alpha_max=config.SMOOTHING_ALPHA_MAX,
        cursor_deadzone=config.CURSOR_DEADZONE,
    )

    ui = UIRenderer()

    window_name = "Invisible Hand Gesture Mouse Controller"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    mirror_feed = not args.no_mirror if args.no_mirror else config.MIRROR_FEED
    is_paused = False
    hand_absent_frames = 0

    # Recording & Snapshot state
    is_recording = bool(args.record)
    video_writer: Optional[cv2.VideoWriter] = None
    record_start_time = time.time() if is_recording else 0.0
    snapshot_flash_timer = 0.0
    output_video_path = args.output_video

    def start_recording(w: int, h: int) -> Optional[cv2.VideoWriter]:
        out_dir = os.path.dirname(output_video_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video_path, fourcc, 30.0, (w, h))
        if writer.isOpened():
            print(f"\n[INFO] Started recording live demo video to: {output_video_path}")
            return writer
        print(f"\n[ERROR] Unable to open VideoWriter for {output_video_path}", file=sys.stderr)
        return None

    def stop_recording(writer: Optional[cv2.VideoWriter]) -> None:
        if writer is not None:
            writer.release()
            print(f"\n[INFO] Recording saved successfully to: {output_video_path}")

    # FPS Calculation tracking
    prev_frame_time = time.time()
    fps_smooth = 60.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARNING] Empty frame received from webcam. Retrying...")
                time.sleep(0.01)
                continue

            # Mirror webcam horizontally for intuitive natural interaction
            if mirror_feed:
                frame = cv2.flip(frame, 1)

            frame_h, frame_w = frame.shape[:2]

            # Lazy initialize video writer if recording was requested via CLI
            if is_recording and video_writer is None:
                video_writer = start_recording(frame_w, frame_h)
                record_start_time = time.time()

            # Track hand landmarks via MediaPipe
            tracker.process_frame(frame)
            landmarks = tracker.get_hand_landmarks(frame.shape, hand_index=0)
            handedness = tracker.get_handedness(hand_index=0)

            # Evaluate gesture classification
            if not is_paused:
                if landmarks is not None:
                    hand_absent_frames = 0
                    gesture_result = detector.update(landmarks, handedness)

                    # Action Dispatch: Move Cursor
                    if gesture_result.gesture == "MOVE" and gesture_result.cursor_pos is not None:
                        cx, cy = gesture_result.cursor_pos
                        mouse.move(cx, cy, frame_w, frame_h)

                    # Action Dispatch: Left Click
                    if gesture_result.click_event:
                        mouse.click()

                    # Action Dispatch: Scroll
                    if gesture_result.gesture == "SCROLL" and abs(gesture_result.scroll_delta) > 0.0:
                        mouse.scroll(gesture_result.scroll_delta)
                else:
                    # Hand is absent from frame
                    hand_absent_frames += 1
                    # Only reset smoothing filter memory if hand has been absent for multiple frames
                    if hand_absent_frames > 8:
                        mouse.reset()
                    gesture_result = detector.update(None)
            else:
                gesture_result = GestureResult(gesture="IDLE")
                mouse.reset()

            # Render UI Overlays
            if config.SHOW_INTERACTION_ZONE:
                ui.draw_interaction_zone(frame)

            if config.SHOW_LANDMARKS and landmarks is not None:
                tracker.draw_landmarks(frame, hand_index=0, draw_connections=config.SHOW_SKELETON)

            ui.draw_finger_highlights(frame, landmarks, gesture_result)

            # Compute real-time smoothed FPS
            curr_frame_time = time.time()
            dt = curr_frame_time - prev_frame_time
            prev_frame_time = curr_frame_time
            if dt > 0.0:
                current_fps = 1.0 / dt
                fps_smooth = 0.9 * fps_smooth + 0.1 * current_fps

            if config.SHOW_HUD_PANEL:
                ui.draw_hud_panel(frame, gesture_result, fps_smooth, is_paused=is_paused)

            record_duration = (time.time() - record_start_time) if is_recording else 0.0
            is_flashing = (time.time() - snapshot_flash_timer) < 1.5
            ui.draw_footer_bar(
                frame,
                is_recording=is_recording,
                record_duration=record_duration,
                snapshot_flash=is_flashing,
            )

            # Write frame to video if recording is active
            if is_recording and video_writer is not None:
                video_writer.write(frame)

            # Display composite window
            cv2.imshow(window_name, frame)

            # Process keyboard interactions
            key = cv2.waitKey(1) & 0xFF
            if key in config.EXIT_KEYS:
                print("\n[INFO] Exit key pressed. Shutting down cleanly...")
                break
            elif key in config.PAUSE_KEYS:
                is_paused = not is_paused
                status_str = "PAUSED" if is_paused else "RESUMED"
                print(f"[INFO] Tracking {status_str}")
            elif key in config.RECORD_KEYS:
                is_recording = not is_recording
                if is_recording:
                    video_writer = start_recording(frame_w, frame_h)
                    record_start_time = time.time()
                else:
                    stop_recording(video_writer)
                    video_writer = None
            elif key in config.SNAPSHOT_KEYS:
                os.makedirs(os.path.dirname(config.DEFAULT_SNAPSHOT_PATH) or ".", exist_ok=True)
                cv2.imwrite(config.DEFAULT_SNAPSHOT_PATH, frame)
                snapshot_flash_timer = time.time()
                print(f"\n[INFO] Live snapshot saved to: {config.DEFAULT_SNAPSHOT_PATH}")

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt received. Exiting...")
    except Exception as exc:
        print(f"\n[ERROR] An unexpected error occurred: {exc}", file=sys.stderr)
        raise
    finally:
        # Graceful cleanup
        if video_writer is not None:
            stop_recording(video_writer)
            video_writer = None
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
        mouse.reset()
        print("[INFO] Resources cleanly released. Goodbye!")


if __name__ == "__main__":
    main()
