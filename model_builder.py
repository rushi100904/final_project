from tensorflow.keras import layers, models
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Lambda

def spectral_indices(x):
    R=x[:,:,:,0]; G=x[:,:,:,1]; B=x[:,:,:,2]
    eps=1e-6
    ndvi=(G-R)/(G+R+eps)
    ndbi=(B-G)/(B+G+eps)
    ndwi=(G-B)/(G+B+eps)
    ndvi=K.mean(ndvi,axis=[1,2])
    ndbi=K.mean(ndbi,axis=[1,2])
    ndwi=K.mean(ndwi,axis=[1,2])
    return K.stack([ndvi,ndbi,ndwi],axis=1)

def build_model():
    inp=layers.Input(shape=(64,64,3))
    x=layers.Conv2D(32,(3,3),activation='relu')(inp)
    x=layers.MaxPooling2D()(x)
    x=layers.Conv2D(64,(3,3),activation='relu')(x)
    x=layers.MaxPooling2D()(x)
    x=layers.Conv2D(128,(3,3),activation='relu')(x)
    x=layers.MaxPooling2D()(x)
    x=layers.Flatten()(x)
    deep=layers.Dense(128,activation='relu')(x)
    spec=Lambda(spectral_indices)(inp)
    comb=layers.Concatenate()([deep,spec])
    comb=layers.Dense(128,activation='relu')(comb)
    comb=layers.Dropout(0.5)(comb)
    out=layers.Dense(10,activation='softmax')(comb)
    return models.Model(inputs=inp,outputs=out)

