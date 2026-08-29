import cv2

from camera.src.camera_detector import CameraDetector


VIDEO_PATH = (
    r"C:\Users\lokit\Desktop\RADIal_data"
    r"\RECORD@2020-11-21_13.44.44"
    r"\RECORD@2020-11-21_13.44.44_preview.avi"
)


def main():
    detector = CameraDetector()

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError("Could not open RADIal video.")

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