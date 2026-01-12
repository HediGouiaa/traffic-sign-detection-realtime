import cv2
import time
from ultralytics import YOLO
import torch

# Choose GPU if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# Load your trained YOLOv8 model
model = YOLO("C:/Users/Gigabyte/Desktop/Nouveau dossier/runs/detect/train2/weights/best.pt")
model.to(DEVICE)

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO inference
    results = model.predict(source=frame, imgsz=640, conf=0.25, device=DEVICE, half=True, verbose=False)
    annotated_frame = results[0].plot()

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(annotated_frame, f"FPS: {int(fps)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # Show live window
    cv2.imshow("YOLOv8 Live", annotated_frame)

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
