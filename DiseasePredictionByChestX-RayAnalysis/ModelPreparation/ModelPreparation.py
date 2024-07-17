import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score

# Import Dataset, split independent and dependent variables
dataset_path = 'C:\\Users\\SMAJUMDAR\\AI_ML_HACKATHON_2024_1\\Hackathon2024_1\\DiseasePredictionByChestX-RayAnalysis\\DataSets\\train_normalised_hot_encoded_data.csv'
#dataset_path = 'C:\\Users\\SMAJUMDAR\\AI_ML_HACKATHON_2024_1\\Hackathon2024_1\\DiseasePredictionByChestX-RayAnalysis\\DataSets\\testing.csv'
dataset = pd.read_csv(dataset_path, header=0)
x = dataset.iloc[:, :-17].values
y = dataset.iloc[:, -15:].values
print("data loaded successfully")

# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.001, random_state = 12)
# print("splitting train test successfully")

# Train the model
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
# will use a MultiOutputClassifier(decision_tree_classifier)
# to make decision tree to be used for multiple output
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 12)
multi_classifier=MultiOutputClassifier(classifier)
multi_classifier.fit(x, y)
print("model trained successfully")

# pred_y = multi_classifier.predict(x_test)
# print(pred_y)
# print(y_test)
# matrix = confusion_matrix(y_test.argmax(axis=1), pred_y.argmax(axis=1))
# print(matrix)
# print(accuracy_score(y_test.argmax(axis=1), pred_y.argmax(axis=1)))

# Save the model
trained_model_dir = 'C:\\Users\\SMAJUMDAR\\AI_ML_HACKATHON_2024_1\\Hackathon2024_1\\DiseasePredictionByChestX-RayAnalysis\\ModelPreparation'
trained_model_file = trained_model_dir + "\\trained_model.sav"
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
