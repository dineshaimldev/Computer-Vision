import cv2
import ultralytics as ul
from collections import defaultdict, deque

# Load YOLO model
model = ul.YOLO("yolov8n.pt")

# Open video
cap = cv2.VideoCapture("video.mp4")

# Check video
if not cap.isOpened():
    print("Error: Could not open video.mp4")
    exit()

# Custom ID mapping
id_map = {}
next_id = 0

# Movement trail
trail = defaultdict(lambda: deque(maxlen=30))

# Appearance count
appear = defaultdict(int)

while True:

    # Read frame
    ret, frame = cap.read()

    if not ret:
        break

    # YOLO tracking
    results = model.track(
        frame,
        classes=[0],
        persist=True,
        verbose=False
    )

    # Copy frame
    annotated_frame = frame.copy()

    # Check tracking IDs
    if results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, oid in zip(boxes, ids):

            # Bounding box
            x1, y1, x2, y2 = map(int, box)

            # Center point
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Count appearances
            appear[oid] += 1

            # Assign custom ID
            if appear[oid] >= 5 and oid not in id_map:
                id_map[oid] = next_id
                next_id += 1

            # Draw if ID exists
            if oid in id_map:

                sid = id_map[oid]

                # Add point to trail
                trail[oid].append((cx, cy))

                # Bounding box
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Person ID
                cv2.putText(
                    annotated_frame,
                    f"Person ID: {sid}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                # Center point
                cv2.circle(
                    annotated_frame,
                    (cx, cy),
                    5,
                    (0, 0, 255),
                    -1
                )

                # Draw trail
                points = list(trail[oid])

                for i in range(1, len(points)):
                    cv2.line(
                        annotated_frame,
                        points[i - 1],
                        points[i],
                        (255, 0, 0),
                        2
                    )

    # Display unique people
    cv2.putText(
        annotated_frame,
        f"Unique People: {next_id}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    # Show video
    cv2.imshow(
        "YOLOv8 Person Tracking",
        annotated_frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release
cap.release()
cv2.destroyAllWindows()