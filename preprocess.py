import cv2
import numpy as np
from PIL import Image
import albumentations as A

# image size according to your model
IMG_SIZE = 64


# ---------------------------
# CLAHE Contrast Enhancement
# ---------------------------
def apply_clahe(image):
    # convert RGB -> LAB
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    cl = clahe.apply(l)

    # merge back
    merged = cv2.merge((cl,a,b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

    return enhanced


# ---------------------------
# Morphological Operations
# ---------------------------
def morphological_operations(image):

    kernel = np.ones((3,3), np.uint8)

    # Erosion
    erosion = cv2.erode(image, kernel, iterations=1)

    # Dilation
    dilation = cv2.dilate(erosion, kernel, iterations=1)

    return dilation


# ---------------------------
# Gaussian Blur (Noise Removal)
# ---------------------------
def remove_noise(image):
    blurred = cv2.GaussianBlur(image, (5,5), 0)
    return blurred


# ---------------------------
# Data Augmentation
# ---------------------------
augmentor = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05,
                       scale_limit=0.1,
                       rotate_limit=20,
                       p=0.5)
])


def augment_image(image):
    augmented = augmentor(image=image)
    return augmented["image"]


# ---------------------------
# MAIN PREPROCESS FUNCTION
# ---------------------------
def preprocess_image(image_path, augment=False):

    # 1) Load image
    img = Image.open(image_path).convert("RGB")
    img = np.array(img)

    # 2) Resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # 3) CLAHE contrast enhancement
    img = apply_clahe(img)

    # 4) Morphological Erosion + Dilation
    img = morphological_operations(img)

    # 5) Gaussian Blur (denoising)
    img = remove_noise(img)

    # 6) Data Augmentation (only during training)
    if augment:
        img = augment_image(img)

    # Save copy for spectral indices (NDVI etc.)
    original_img = img.copy()

    # 7) Normalization (0 → 1)
    img = img.astype("float32") / 255.0

    # expand dimension for CNN input
    img = np.expand_dims(img, axis=0)

    return img, original_img