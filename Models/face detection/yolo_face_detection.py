import cv2
from ultralytics import YOLO

# Load YOLO face detection model (yolov8n-face.pt is a face-specific model)
# Using YOLOv8 nano model and filtering for faces
model = YOLO('yolov8n.pt')

# Open camera with AVFOUNDATION backend
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("Press 'q' to quit")
print("Loading YOLO model...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame")
        break

    # Run YOLO inference
    results = model(frame, verbose=False)

    # Draw face bounding boxes
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            
            # Only draw high confidence detections
            if conf > 0.5:
                # Draw rectangle
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Add confidence label
                label = f"Face {conf:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('YOLO Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()