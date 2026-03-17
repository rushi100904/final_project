from flask import Flask, render_template, request
import os
import numpy as np
from PIL import Image
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Lambda
import tensorflow.keras.backend as K

# =========================
# Flask App Setup
# =========================
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

IMG_SIZE = 64

# =========================
# Spectral Indices Function
# =========================
def spectral_indices(x):
    R = x[:, :, :, 0]
    G = x[:, :, :, 1]
    B = x[:, :, :, 2]
    eps = 1e-6

    ndvi = (G - R) / (G + R + eps)
    ndbi = (B - G) / (B + G + eps)
    ndwi = (G - B) / (G + B + eps)

    ndvi = K.mean(ndvi, axis=[1, 2])
    ndbi = K.mean(ndbi, axis=[1, 2])
    ndwi = K.mean(ndwi, axis=[1, 2])

    return K.stack([ndvi, ndbi, ndwi], axis=1)

# =========================
# Build IHFEC Model
# =========================
def build_model():
    inp = layers.Input(shape=(64, 64, 3))

    x = layers.Conv2D(32, (3,3), activation='relu')(inp)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, (3,3), activation='relu')(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, (3,3), activation='relu')(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Flatten()(x)
    deep = layers.Dense(128, activation='relu')(x)

    spec = Lambda(spectral_indices)(inp)

    comb = layers.Concatenate()([deep, spec])
    comb = layers.Dense(128, activation='relu')(comb)
    comb = layers.Dropout(0.5)(comb)

    out = layers.Dense(10, activation='softmax')(comb)

    return models.Model(inputs=inp, outputs=out)

# =========================
# Load Model
# =========================
model = build_model()
model.load_weights("model/ihfec.weights.h5")

# EuroSAT Classes
class_names = [
    'AnnualCrop','Forest','HerbaceousVegetation','Highway','Industrial',
    'Pasture','PermanentCrop','Residential','River','SeaLake'
]

# =========================
# 3 Category Mapping
# =========================
category_map = {

    "Forest": "Forest Areas",

    "River": "Water Bodies",
    "SeaLake": "Water Bodies",

    "Residential": "Urban Areas",
    "Industrial": "Urban Areas",
    "Highway": "Urban Areas",

    "AnnualCrop": "Forest Areas",
    "HerbaceousVegetation": "Forest Areas",
    "Pasture": "Forest Areas",
    "PermanentCrop": "Forest Areas"
}

# =========================
# Preprocessing Function
# =========================
def preprocess_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img).astype("float32") / 255.0
    original = img.copy()
    img = np.expand_dims(img, axis=0)
    return img, original

# =========================
# Severity Calculation
# =========================
def calculate_severity(image):
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]
    eps = 1e-6

    ndvi = (G - R) / (G + R + eps)
    ndbi = (B - G) / (B + G + eps)
    ndwi = (G - B) / (G + B + eps)

    ndvi = np.mean(ndvi)
    ndbi = np.mean(ndbi)
    ndwi = np.mean(ndwi)

    if ndvi > 0.25 and ndwi > 0.1 and ndbi < 0:
        severity = "LOW"
    elif ndbi > 0.15 and ndvi < 0.2:
        severity = "SEVERE"
    else:
        severity = "MODERATE"

    return severity, ndvi, ndbi, ndwi

# =========================
# Prediction Function
# =========================
def predict_image(path):
    img, orig = preprocess_image(path)

    pred = model.predict(img)
    idx = np.argmax(pred)
    conf = float(np.max(pred))

    original_label = class_names[idx]

    # Convert to 3 categories
    label = category_map.get(original_label, "Forest Areas")

    sev, ndvi, ndbi, ndwi = calculate_severity(orig)

    return label, conf, sev, ndvi, ndbi, ndwi

# =========================
# Flask Routes
# =========================
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    file = request.files['image']
    path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(path)

    label, conf, sev, ndvi, ndbi, ndwi = predict_image(path)

    return render_template(
        "result.html",
        label=label,
        confidence=round(conf * 100, 2),
        severity=sev,
        ndvi=round(ndvi, 3),
        ndbi=round(ndbi, 3),
        ndwi=round(ndwi, 3),
        image_path=path
    )

# =========================
# Run App
# =========================
if __name__ == "__main__":
    app.run(debug=True)