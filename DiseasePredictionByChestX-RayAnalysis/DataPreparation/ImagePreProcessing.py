import numpy as np
import pandas as pd
import os
from DataNormalizationFromImage import normalizeImage


train_data_path = "C:\\Users\\SMAJUMDAR\\AI_ML_HACKATHON_2024_1_DATASET\\train_image_disease_map.csv"
train_dataset = pd.read_csv(train_data_path, header=0, usecols=["Image Index","Finding Labels"])

train_file_name_list=list(train_dataset["Image Index"])
train_target_disease_column_list=list(train_dataset["Finding Labels"])

train_image_dir = "C:\\Users\\SMAJUMDAR\\AI_ML_HACKATHON_2024_1_DATASET\\train_images"
#train_image_dir = "C:\\Users\\SMAJUMDAR\\AI_ML_HACKATHON_2024_1_DATASET\\sampoo_tests"

image_count=0
normalized_image_data_list = np.array([])
for dirpath, dirnames, file_name_list in os.walk(train_image_dir):
    for file_name in file_name_list:
        image_count=image_count+1
        normalized_image_data_list=np.append(normalized_image_data_list,normalizeImage(dirpath+"//"+file_name))
        file_index=train_file_name_list.index(file_name)
        normalized_image_data_list=np.append(normalized_image_data_list,train_target_disease_column_list[file_index])
        if image_count%500==0:
            print(image_count)

#Reshaping the single array into a 2d array
normalized_data_list = normalized_image_data_list.reshape(-1,4097)

#Save the normalized data as csv file
df = pd.DataFrame(normalized_data_list)
normalized_data_csv_path = 'C:\\Users\\SMAJUMDAR\\AI_ML_HACKATHON_2024_1\\Hackathon2024_1\\DiseasePredictionByChestX-RayAnalysis\\DataSets\\train_normalised_data.csv'
df.to_csv(normalized_data_csv_path, index=False)
print("Done...")
# normalized_data_list = np.array([])
# for i in range(0,2):
#     imagePath = ".//ImageProcessingPOC//00000001_00"+str(i)+".png"
#     normalized_data_list = np.append(normalized_data_list, normalizeImage(imagePath))

# #Reshaping the single array into a 2d array
# normalized_data_list = normalized_data_list.reshape(-1,65)

# #Save the normalized data as csv file
# df = pd.DataFrame(normalized_data_list)
# df.to_csv('./Datasets/ChestX-RayData/Prepared-Data.csv')