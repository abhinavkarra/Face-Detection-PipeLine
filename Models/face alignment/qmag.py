import cv2
from ultralytics import YOLO
import torch
# Assuming QMagFace models are loaded via your cloned repo
# from model import QMagFace 

# 1. Load YOLO (Detector)
detector = YOLO('yolov8n.pt')

# 2. Load QMagFace (Recognizer)
# model = load_qmagface_model('path_to_weights.pth')
# model.eval()

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

while True:
    ret, frame = cap.read()
    if not ret: break

    # --- DETECTION PHASE ---
    results = detector(frame, stream=True)
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # --- PRE-PROCESSING ---
            # Crop the face found by YOLO
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size > 0:
                # QMagFace alignment/resize usually happens here
                # aligned_face = align_face(face_crop) 
                
                # --- RECOGNITION PHASE ---
                # with torch.no_grad():
                #    embedding, quality = model(aligned_face)
                
                # Draw bounding box and Quality Score
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"Quality: Processing...", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow('YOLO + QMagFace', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()