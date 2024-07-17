import numpy as np
import pandas as pd
import os
from PIL import Image
from DataNormalizationFromImage import normalizeImage

def one_hot_encoding(df):
    # df represents the dataframe
    # column no. 1024 holds the target diseases appended with pipe ('|')
    df['targets_split'] = df[2025].str.split('|')

    # all 15 target diseases 
    unique_targets = ['Atelectasis','Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis', 'Effusion',
                         'Pneumonia', 'Pleural_Thickening', 'Cardiomegaly', 'Nodule' , 'Mass', 'Hernia', 'No Finding']

    # Create binary columns
    # Doing the one hot encoding for 15 disease labels
    for target in unique_targets:
        df[target] = df['targets_split'].apply(lambda x: 1 if target in x else 0)

    return df


# train image names to disease name mapping file : train_image_disease_map.csv 
train_data_path = "C:\\Users\\SMAJUMDAR\\AI_ML_HACKATHON_2024_1_DATASET\\train_image_disease_map.csv"
# reading only two columns from the csv, keeping header as the 0th row
train_dataset = pd.read_csv(train_data_path, header=0, usecols=["Image Index","Finding Labels"])

# List of the Image names
train_file_name_list=list(train_dataset["Image Index"])
# creatng map/dictionary as <image_name : image_index>
train_file_name_map = { train_file_name_list[i]: i for i in range(0,len(train_file_name_list))}
#List of Target Disease for each row
train_target_disease_column_list=list(train_dataset["Finding Labels"])

# train images directory : train_images
train_image_dir = "C:\\Users\\SMAJUMDAR\\AI_ML_HACKATHON_2024_1_DATASET\\train_images"

image_count = 0
# after each image_count_threshold value, csv file wil be appended with data
image_count_threshold = 100
# the flatten image data length as 32*32=1024 /45*45=2025
data_length = 2025
# while declaring np.array, first row we need to declare to represent size with all zeros
normalized_image_data_list = np.array([[0 for i in range(0, data_length+1)]])
for dirpath, dirnames, file_name_list in os.walk(train_image_dir):
    for file_name in file_name_list:
        image_count = image_count+1

        # Opening the image from image path
        image=Image.open(dirpath+"\\"+file_name)
        #print(image)

        # declaring an empty numpy array for single image
        normalized_image = np.array([])

        # noralishing the image and appendin it into the array
        normalized_image = np.append(normalized_image,normalizeImage(image))

        # normalized_image_data_list=np.append(normalized_image_data_list,normalizeImage(dirpath+"//"+file_name))
        # fetching image index from image name from dictionary
        file_index=train_file_name_map[file_name]

        # Appending the target/disease labels into the array
        normalized_image=np.append(normalized_image, train_target_disease_column_list[file_index])

        # Adding single image arra into the 2-d np array
        normalized_image_data_list = np.vstack([normalized_image_data_list, normalized_image])

        # for each threshold value occurring, we'll save the normalised data in csv
        if image_count%image_count_threshold==0:
            # Do one hot coding for the target labels
            # create dataframe from the np.array
            # exclude the first row consisting with only zeros (0's)
            df = one_hot_encoding(pd.DataFrame(normalized_image_data_list[1:]))
            
            # normalised data path
            normalized_data_csv_path = 'C:\\Users\\SMAJUMDAR\\AI_ML_HACKATHON_2024_1\\Hackathon2024_1\\DiseasePredictionByChestX-RayAnalysis\\DataSets\\train_normalised_hot_encoded_data_new_45.csv'
            
            if image_count_threshold == image_count:
                df.to_csv(normalized_data_csv_path, index=False, mode='w')
            else:
                df.to_csv(normalized_data_csv_path, index=False, mode='a', header=False)

            # Re-Initializing 2-d np.array() again with zeros (0's)
            normalized_image_data_list = np.array([[0 for i in range(0, data_length+1)]])
            print("Done:", image_count)

#Reshaping the single array into a 2d array
#normalized_data_list = normalized_image_data_list.reshape(-1,16385)

#Save the normalized data as csv file
df = one_hot_encoding(pd.DataFrame(normalized_image_data_list[1:]))
normalized_data_csv_path = 'C:\\Users\\SMAJUMDAR\\AI_ML_HACKATHON_2024_1\\Hackathon2024_1\\DiseasePredictionByChestX-RayAnalysis\\DataSets\\train_normalised_hot_encoded_data_new_45.csv'
df.to_csv(normalized_data_csv_path, index=False, mode='a', header=False)

print("Done...")

# import datetime
# import pandas as pd

# train_data_path = "D:\\Datasets\\X-Ray_Dataset\\Data_Entry_2017.csv"
# train_dataset = pd.read_csv(train_data_path, header=0, usecols=["Image Index","Finding Labels"])
# train_file_name_list=list(train_dataset["Image Index"])
# dt1 = datetime.datetime.now()
# file_index=train_file_name_list.index("00030805_000.png")
# dt2 = datetime.datetime.now()
# print(((dt2.microsecond) - (dt1.microsecond))/1000)
# print(file_index)


# normalized_data_list = np.array([])
# for i in range(0,2):
#     imagePath = ".//ImageProcessingPOC//00000001_00"+str(i)+".png"
#     normalized_data_list = np.append(normalized_data_list, normalizeImage(imagePath))

# #Reshaping the single array into a 2d array
# normalized_data_list = normalized_data_list.reshape(-1,65)

# #Save the normalized data as csv file
# df = pd.DataFrame(normalized_data_list)
# df.to_csv('./Datasets/ChestX-RayData/Prepared-Data.csv')