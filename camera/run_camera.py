import argparse
from pathlib import Path
import cv2

from camera.src.camera_detector import CameraDetector
from config import get_default_recording_dir, find_video_file


def main():
    parser = argparse.ArgumentParser(description="RADIal Camera Object Detection")
    parser.add_argument(
        "--video",
        default=None,
        help="Path to camera video file (*preview.avi, *.avi, *.mp4).",
    )
    parser.add_argument(
        "--recording",
        default=None,
        help="Path to RADIal recording folder.",
    )
    args = parser.parse_args()

    if args.video:
        video_path = Path(args.video).resolve()
    else:
        rec_dir = Path(args.recording).resolve() if args.recording else get_default_recording_dir()
        video_path = find_video_file(rec_dir)

    detector = CameraDetector()

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open RADIal video: {video_path}")

    print("Starting camera object detection...")
    print("Press Q to quit.")

    frame_number = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        annotated_frame, detections = detector.detect(frame)

        print(
            f"Frame {frame_number}: "
            f"{len(detections)} objects detected"
        )

        cv2.imshow(
            "RADIal Camera - YOLO Detection",
            annotated_frame,
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("Camera detection finished.")


if __name__ == "__main__":
    main()