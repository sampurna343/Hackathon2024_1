from PIL import Image, ImageFilter
import numpy as np
import pandas as pd
import os


def normalizeImage(imagePath):
    #Opening the Image
    img = Image.open(imagePath)
    
    #Converting from png to RGB
    #img = img.convert('RGB')
    #No point in conversion to RGB as this are grayscaled image (149,149,149)

    #get the dimension of the Image
    print(img.getbbox())

    #get band names ('R','G','B')
    print(img.getbands())

    #Filtering the Image
    #https://pillow.readthedocs.io/en/stable/reference/ImageFilter.html
    img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)

    #Resizing the Image into lower pixels
    img = img.resize((8,8))

    #Thresholding the Image
    img = img.convert("1", None, None)

    #Fetching image data in flattened way
    pixels = img.getdata()
    #print(list(pixels))

    #Convert to NP array and normalize the values
    normalized_data = np.array(list(pixels))//255
    #normalized_data = np.append(normalized_data,"Cardiomegaly")
    #print(normalized_data)
    return normalized_data

    # # By reshaping we are making it a 2d array 
    # # with 1 row and 65 columns
    # normalized_data = normalized_data.reshape(1,65)
    # print(normalized_data)

    # #Save the normalized data as csv file
    # df = pd.DataFrame(normalized_data)
    # df.to_csv('./Datasets/ChestX-RayData/Prepared-Data.csv')

    # data = pd.read_csv('./Datasets/ChestX-RayData/Prepared-Data.csv')

    # #Show the Image
    # img.show()

train_data_path = "C:\\Users\\SMAJUMDAR\\AI_ML_HACKATHON_2024_1_DATASET\\train_image_disease_map.csv"
train_dataset = pd.read_csv(train_data_path, header=0)

train_file_name_list=train_dataset[0]
train_target_disease_column_list=train_dataset[-1]

#train_image_dir = "C:\\Users\\SMAJUMDAR\\AI_ML_HACKATHON_2024_1_DATASET\\train_images"
train_image_dir = "C:\\Users\\SMAJUMDAR\\AI_ML_HACKATHON_2024_1_DATASET\\sampoo_tests"

normalized_image_data_list = np.array([])
for dirpath, dirnames, file_name_list in os.walk(train_image_dir):
    for file_name in file_name_list:
        normalized_image_data_list=np.append(normalized_image_data_list,normalizeImage(dirpath+"//"+file_name))
        file_index=train_file_name_list.index(file_name)
        normalized_image_data_list=np.append(normalized_image_data_list,train_target_disease_column_list[file_index])

#Reshaping the single array into a 2d array
normalized_data_list = normalized_image_data_list.reshape(-1,65)

#Save the normalized data as csv file
df = pd.DataFrame(normalized_data_list)
df.to_csv('C:\\Users\\SMAJUMDAR\\AI_ML_HACKATHON_2024_1\\Hackathon2024_1\\DiseasePredictionByChestX-RayAnalysis\\DataSets\\train_normalised_data_dummy.csv')

# normalized_data_list = np.array([])
# for i in range(0,2):
#     imagePath = ".//ImageProcessingPOC//00000001_00"+str(i)+".png"
#     normalized_data_list = np.append(normalized_data_list, normalizeImage(imagePath))

# #Reshaping the single array into a 2d array
# normalized_data_list = normalized_data_list.reshape(-1,65)

# #Save the normalized data as csv file
# df = pd.DataFrame(normalized_data_list)
# df.to_csv('./Datasets/ChestX-RayData/Prepared-Data.csv')