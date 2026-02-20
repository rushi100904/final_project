import numpy as np

def calculate_indices(image):

    R = image[:,:,0]
    G = image[:,:,1]
    B = image[:,:,2]

    eps = 1e-6

    # NDVI (vegetation)
    NDVI = (G - R) / (G + R + eps)

    # NDBI (urban)
    NDBI = (B - G) / (B + G + eps)

    # NDWI (water)
    NDWI = (G - B) / (G + B + eps)

    ndvi_mean = np.mean(NDVI)
    ndbi_mean = np.mean(NDBI)
    ndwi_mean = np.mean(NDWI)

    return ndvi_mean, ndbi_mean, ndwi_mean


def calculate_severity(image):

    ndvi, ndbi, ndwi = calculate_indices(image)

    # Severity Logic
    if ndvi > 0.25 and ndwi > 0.1 and ndbi < 0:
        severity = "LOW"
    elif ndbi > 0.15 and ndvi < 0.2:
        severity = "SEVERE"
    else:
        severity = "MODERATE"

    return severity, ndvi, ndbi, ndwi
