import os
from skimage import io
import numpy as np
from PIL import Image
import pandas as pd

result = np.array([])
for dirpath, dirnames, files in os.walk('C:\\Users\\SMAJUMDAR\\X-Ray Images\\hello'):
    print(f'Found directory: {dirpath}')
    for file in files:
        print(file)
        image = io.imread(dirpath+"\\"+file)
        img_pil = Image.fromarray(image)
        img_arr = np.array(img_pil.resize((64,64)))
        img_arr = img_arr.flatten()
        print(img_arr)
        print(len(img_arr))
        result = np.vstack([result, img_arr])
        print(len(result))
print(result)
df = pd.DataFrame(result)
df.to_csv('./Datasets/ChestX-RayData/Prepared-Data.csv')


# image = io.imread("./ImageProcessing/World-Covid-Map.jpeg")
# img_pil = Image.fromarray(image)
# img_arr = np.array(img_pil.resize((400,280)))
# #img_arr = img_arr.flatten()
# #img_arr = np.append(img_arr, "Heart Attack")
# #print(img_arr)

# import matplotlib.pyplot as plt
# plt.imshow(img_arr)
# plt.show()
