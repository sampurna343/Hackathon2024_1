import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil

excel_dir = "C:\\Users\\SMAJUMDAR\\AI_ML_HACKATHON_2024_1_DATASET\\image_disease_map.csv"
dataset = pd.read_csv(excel_dir, header=0)
header_list = dataset.columns.tolist()
print("data loaded successfully")

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.01, random_state = 12)

train_data = pd.DataFrame(x_train)
train_data.insert(10, 10, y_train, True)
# print(train_data)
train_data_path = "C:\\Users\\SMAJUMDAR\\AI_ML_HACKATHON_2024_1_DATASET\\train_image_disease_map.csv"
train_data.to_csv(train_data_path, index=False, header=header_list)
print("trained data saved successfully")

test_data = pd.DataFrame(x_test)
test_data.insert(10, 10, y_test, True)
# print(test_data)
test_data_path = "C:\\Users\\SMAJUMDAR\\AI_ML_HACKATHON_2024_1_DATASET\\test_image_disease_map.csv"
test_data.to_csv(test_data_path, index=False, header=header_list)
print("test data saved successfully")

train_image_names = list(train_data[0])
test_image_names = list(test_data[0])

all_image_dir = "C:\\Users\\SMAJUMDAR\\AI_ML_HACKATHON_2024_1_DATASET\\all_images"
train_image_dir = "C:\\Users\\SMAJUMDAR\\AI_ML_HACKATHON_2024_1_DATASET\\train_images"
test_image_dir = "C:\\Users\\SMAJUMDAR\\AI_ML_HACKATHON_2024_1_DATASET\\test_images"

image_count=0
print("starting image copying")

for dirpath, dirnames, file_name_list in os.walk(all_image_dir):
    #print("Directory Path :", dirpath)
    for file_name in file_name_list:
        #print("File Name : ", file_name)
        image_count=image_count+1
        if file_name in test_image_names:
            shutil.copyfile(dirpath+"//"+file_name,test_image_dir+"//"+file_name)
        else:
            shutil.copyfile(dirpath+"//"+file_name,train_image_dir+"//"+file_name)

        if image_count%500==0:
            print(image_count)
print("Done")


