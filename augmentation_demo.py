import cv2
import numpy as np
import random

def apply_4_augmentations(image):
    """
    Apply 4 random augmentations to an image
    """
    augs = []
    
    # 1. Brightness adjustment
    brightness = random.uniform(0.7, 1.3)
    aug1 = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
    augs.append(('Brightness', aug1))
    
    # 2. Contrast adjustment
    contrast = random.uniform(0.7, 1.3)
    aug2 = cv2.convertScaleAbs(image, alpha=contrast, beta=0)
    augs.append(('Contrast', aug2))
    
    # 3. Gaussian Blur
    kernel_size = random.choice([3, 5])
    aug3 = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    augs.append(('Blur', aug3))
    
    # 4. Horizontal Flip
    aug4 = cv2.flip(image, 1)
    augs.append(('Flip', aug4))
    
    return augs

# Load sample image
image_path = 'test_image.jpg'  # Replace with your image path
frame = cv2.imread(image_path)

if frame is None:
    print("Creating sample image...")
    # Create a sample image if none provided
    frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

print("Original image shape:", frame.shape)

# Apply 4 augmentations
augmented_images = apply_4_augmentations(frame)

# Display results
for name, img in augmented_images:
    print(f"{name}: shape = {img.shape}")

# Save augmented images
for i, (name, img) in enumerate(augmented_images):
    cv2.imwrite(f"augmented_{i+1}_{name.lower()}.jpg", img)
    print(f"Saved: augmented_{i+1}_{name.lower()}.jpg")

print("\n4 augmentations applied successfully!")