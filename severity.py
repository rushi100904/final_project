import numpy as np

def calculate_severity(image):
    R=image[:,:,0]; G=image[:,:,1]; B=image[:,:,2]
    eps=1e-6

    ndvi=(G-R)/(G+R+eps)
    ndbi=(B-G)/(B+G+eps)
    ndwi=(G-B)/(G+B+eps)

    ndvi=np.mean(ndvi)
    ndbi=np.mean(ndbi)
    ndwi=np.mean(ndwi)

    if ndvi>0.25 and ndwi>0.1 and ndbi<0:
        sev="LOW"
    elif ndbi>0.15 and ndvi<0.2:
        sev="SEVERE"
    else:
        sev="MODERATE"

    return sev,ndvi,ndbi,ndwi