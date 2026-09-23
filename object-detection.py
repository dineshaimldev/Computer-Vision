import cv2
from ultralytics import YOLO

print("Loading YOLO model...")

model = YOLO("yolov8n.pt")

print("YOLO model loaded")
print("Starting camera...")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read camera")
        break

    results = model(frame)

    annotated_frame = results[0].plot()

    cv2.imshow("YOLOv8 Object Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()