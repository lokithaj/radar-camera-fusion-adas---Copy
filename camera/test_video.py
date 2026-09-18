import argparse
from pathlib import Path
import cv2

from config import get_default_recording_dir, find_video_file

def main():
    parser = argparse.ArgumentParser(description="Test RADIal Video Playback")
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

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open RADIal video: {video_path}")

    print("Video opened successfully.")

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        cv2.imshow("RADIal Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("Frames read:", frame_count)


if __name__ == "__main__":
    main()