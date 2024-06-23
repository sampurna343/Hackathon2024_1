from skimage import io
import numpy as np
from PIL import Image

image = io.imread("./ImageProcessing/World-Covid-Map.jpeg")
img_pil = Image.fromarray(image)
img_arr = np.array(img_pil.resize((400,280)))
#img_arr = img_arr.flatten()
#img_arr = np.append(img_arr, "Heart Attack")
#print(img_arr)

import matplotlib.pyplot as plt
plt.imshow(img_arr)
plt.show()
