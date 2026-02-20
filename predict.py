import tensorflow as tf
import numpy as np
from preprocess import preprocess_image
from severity import calculate_severity
from model_builder import build_model

# rebuild architecture locally
model = build_model()

# load trained knowledge (weights)
model.load_weights("model/ihfec.weights.h5")

class_names = [
    'AnnualCrop','Forest','HerbaceousVegetation','Highway','Industrial',
    'Pasture','PermanentCrop','Residential','River','SeaLake'
]

def predict_image(image_path):

    img, original_img = preprocess_image(image_path)

    prediction = model.predict(img)

    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction))
    label = class_names[class_index]

    severity, ndvi, ndbi, ndwi = calculate_severity(original_img)

    return label, confidence, severity, ndvi, ndbi, ndwi

