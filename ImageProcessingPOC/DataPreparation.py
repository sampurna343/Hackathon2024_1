from PIL import Image
import numpy as np

img = Image.open('.//ImageProcessingPOC//00000001_000.png')
#img = Image.open('.//ImageProcessingPOC//World-Covid-Map.jpeg')
img = np.asarray(img)
#img = img.resize(24,24)
img.show()
