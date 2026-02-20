from PIL import Image
import numpy as np

IMG_SIZE=64

def preprocess_image(path):
    img=Image.open(path).convert("RGB")
    img=img.resize((IMG_SIZE,IMG_SIZE))
    img=np.array(img).astype("float32")/255.0
    original=img.copy()
    img=np.expand_dims(img,axis=0)
    return img,original