import numpy as np
from preprocess import preprocess_image
from severity import calculate_severity
from model_builder import build_model

model=build_model()
model.load_weights("model/ihfec.weights.h5")

class_names=['AnnualCrop','Forest','HerbaceousVegetation','Highway','Industrial',
             'Pasture','PermanentCrop','Residential','River','SeaLake']

def predict_image(path):
    img,orig=preprocess_image(path)
    pred=model.predict(img)
    idx=np.argmax(pred)
    conf=float(np.max(pred))
    label=class_names[idx]
    sev,ndvi,ndbi,ndwi=calculate_severity(orig)
    return label,conf,sev,ndvi,ndbi,ndwi

