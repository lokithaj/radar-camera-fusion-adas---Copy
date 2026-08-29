import cv2


VIDEO_PATH = (
    r"C:\Users\lokit\Desktop\RADIal_data"
    r"\RECORD@2020-11-21_13.44.44"
    r"\RECORD@2020-11-21_13.44.44_preview.avi"
)


cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Could not open RADIal video.")

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