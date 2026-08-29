from ultralytics import YOLO


class CameraDetector:
    """YOLO-based camera object detector."""

    def __init__(
        self,
        model_name="yolo11n.pt",
        confidence=0.4,
    ):
        self.model = YOLO(model_name)
        self.confidence = confidence

    def detect(self, frame):
        """Run object detection on one camera frame."""

        results = self.model(
            frame,
            conf=self.confidence,
            verbose=False,
        )

        result = results[0]

        annotated_frame = result.plot()

        detections = []

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            confidence = float(box.conf[0])
            class_id = int(box.cls[0])

            class_name = self.model.names[class_id]

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": [
                        x1,
                        y1,
                        x2,
                        y2,
                    ],
                }
            )

        return annotated_frame, detections