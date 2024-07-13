import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil

excel_dir = "D:\\Datasets\\Demo-X-Ray_Dataset\\Data_Entry.csv"
dataset = pd.read_csv(excel_dir, header=0)
header_list = dataset.columns.tolist()

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 12)

train_data = pd.DataFrame(x_train)
train_data.insert(10, 10, y_train, True)
# print(train_data)
train_data_path = "D:\\Datasets\\Demo-X-Ray_Dataset\\Train_Data_Entry.csv"
train_data.to_csv(train_data_path, index=False, header=header_list)

test_data = pd.DataFrame(x_test)
test_data.insert(10, 10, y_test, True)
# print(test_data)
test_data_path = "D:\\Datasets\\Demo-X-Ray_Dataset\\Test_Data_Entry.csv"
test_data.to_csv(test_data_path, index=False, header=header_list)

train_image_names = list(train_data[0])
test_image_names = list(test_data[0])

image_dir = "D:\\Datasets\\Demo-X-Ray_Dataset\\Images"
train_image_dir = "D:\\Datasets\\Demo-X-Ray_Dataset\\Train_Images"
test_image_dir = "D:\\Datasets\\Demo-X-Ray_Dataset\\Test_Images"
for dirpath, dirnames, files in os.walk(image_dir):
    print("Directory Path :", dirpath)
    for file in files:
        print("File Name : ", file)
        if file in test_image_names:
            shutil.copyfile(dirpath+"//"+file,test_image_dir+"//"+file)
        else:
            shutil.copyfile(dirpath+"//"+file,train_image_dir+"//"+file)
print("Done")


