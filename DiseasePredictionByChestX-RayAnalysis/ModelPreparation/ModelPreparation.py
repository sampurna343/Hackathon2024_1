import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score

# Path of Normalised - one hot coding dataset
dataset_path = 'C:\\Users\\SMAJUMDAR\\AI_ML_HACKATHON_2024_1\\Hackathon2024_1\\DiseasePredictionByChestX-RayAnalysis\\DataSets\\train_normalised_hot_encoded_data_new_45.csv'
#dataset_path = 'C:\\Users\\SMAJUMDAR\\AI_ML_HACKATHON_2024_1\\Hackathon2024_1\\DiseasePredictionByChestX-RayAnalysis\\DataSets\\testing.csv'
dataset = pd.read_csv(dataset_path, header=0)

# -17 is because of two extra target_labels, target_split_labels along with 15 disease count
x = dataset.iloc[:, :-17].values
# -15 is for 15 disease labels
y = dataset.iloc[:, -15:].values
print("data loaded successfully")

# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 12)
# print("splitting train test successfully")

# Train the model
#from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

#classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 12)
classifier = RandomForestClassifier(n_estimators = 10,criterion = 'entropy', random_state = 0)

# to make decision tree to be used for multiple output for train
# multi_classifier=MultiOutputClassifier(classifier)
# multi_classifier.fit(x_train, y_train)
# print("model trained successfully")


# # to make decision tree to be used for multiple output
multi_classifier=MultiOutputClassifier(classifier)
multi_classifier.fit(x, y)
print("model trained successfully")

# pred_y = multi_classifier.predict(x_test)
# # print(pred_y)
# # print(y_test)
# matrix = confusion_matrix(y_test.argmax(axis=1), pred_y.argmax(axis=1))
# # print(matrix)
# print(accuracy_score(y_test.argmax(axis=1), pred_y.argmax(axis=1)))

# Save the model
trained_model_dir = 'C:\\Users\\SMAJUMDAR\\AI_ML_HACKATHON_2024_1\\Hackathon2024_1\\DiseasePredictionByChestX-RayAnalysis\\ModelPreparation'
trained_model_file = trained_model_dir + "\\trained_model_random_forest_45.sav"
pickle.dump(multi_classifier, open(trained_model_file, 'wb'))
print("model saved successfully")

# Load the Model
#trained_model = pickle.load(open(filename,'rb'))

# Prepare Test Data
# from DataPreparation import normalizeImage
# testImagePath = ".//ImageProcessingPOC//00000001_002.png"
# test_data = pd.DataFrame(normalizeImage(testImagePath))
# x_test = test_data

# Predict test data result
# y_pred = trained_model.predict(x_test)
# print(y_pred)
