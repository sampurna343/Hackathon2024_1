from DataPreparation.DataNormalizationFromImage import normalizeImage

def submit(image):
    print(image)
    ok=normalizeImage(image)
    print(ok)