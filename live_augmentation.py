import cv2
import numpy as np
import random

def apply_augmentations(frame):
    """
    Apply 4 augmentations to live webcam frame
    """
    augs = []
    
    # 1. Brightness adjustment
    brightness = random.uniform(0.7, 1.3)
    aug1 = cv2.convertScaleAbs(frame, alpha=brightness, beta=0)
    augs.append(('Brightness', aug1))
    
    # 2. Contrast adjustment
    contrast = random.uniform(0.7, 1.3)
    aug2 = cv2.convertScaleAbs(frame, alpha=contrast, beta=0)
    augs.append(('Contrast', aug2))
    
    # 3. Gaussian Blur
    kernel_size = random.choice([3, 5])
    aug3 = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
    augs.append(('Blur', aug3))
    
    # 4. Horizontal Flip
    aug4 = cv2.flip(frame, 1)
    augs.append(('Flip', aug4))
    
    return augs

# Open camera with AVFOUNDATION backend
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("Press 'q' to quit")
print("Showing original + 4 augmentations")

# Create display window
cv2.namedWindow('Augmentation Demo', cv2.WINDOW_NORMAL)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame")
        break
    
    # Apply augmentations every 30 frames for stability
    if frame_count % 30 == 0:
        augs = apply_augmentations(frame)
    
    # Create a grid layout (2 rows x 3 columns)
    # Original + 4 augmentations
    h, w = frame.shape[:2]
    
    # Resize all to same size
    display_w = w // 2
    display_h = h // 2
    
    # Original frame
    orig_resized = cv2.resize(frame, (display_w, display_h))
    
    # Augmented frames
    aug1 = cv2.resize(augs[0][1], (display_w, display_h))
    aug2 = cv2.resize(augs[1][1], (display_w, display_h))
    aug3 = cv2.resize(augs[2][1], (display_w, display_h))
    aug4 = cv2.resize(augs[3][1], (display_w, display_h))
    
    # Top row: Original + Brightness
    top_row = np.hstack([orig_resized, aug1])
    
    # Bottom row: Contrast + Blur + Flip
    bottom_row = np.hstack([aug2, aug3])
    bottom_row2 = np.hstack([aug4, np.zeros_like(aug4)])
    
    # Combine
    combined = np.vstack([top_row, bottom_row])
    
    # Add labels
    cv2.putText(combined, 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(combined, 'Brightness', (display_w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(combined, 'Contrast', (10, display_h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(combined, 'Blur', (display_w + 10, display_h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Augmentation Demo', combined)
    
    frame_count += 1
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()