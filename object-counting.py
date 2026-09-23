import cv2
from ultralytics import YOLO
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("video.mp4")

unique_ids = set()

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to read video or video finished")
        break

    results = model.track(
        frame,
        classes=[39],
        persist=True,
        verbose=False
    )

    annotated_frame = results[0].plot()

    if results[0].boxes.id is not None:

        ids = results[0].boxes.id.cpu().numpy()

        for oid in ids:
            unique_ids.add(int(oid))

    cv2.putText(
        annotated_frame,
        f"Unique Objects: {len(unique_ids)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Object Counting", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()