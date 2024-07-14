import pandas as pd
import pickle

# Import Dataset, split independent and dependent variables
dataset = pd.read_csv('./Datasets/ChestX-RayData/Prepared-Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.5, random_state = 1)


# Train the model
from sklearn.tree import DecisionTreeClassifier
# from sklearn.multioutput import MultiOutputClassifier
# will use a MultiOutputClassifier(decision_tree_classifier)
# to make decision tree to be used for multiple output
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 12)
classifier.fit(x_train, y_train)

# Save the model
filename = "./Datasets/ChestX-RayData/Trained-Decision-Tree-Model.sav"
pickle.dump(classifier, open(filename, 'wb'))

# Load the Model
trained_model = pickle.load(open(filename,'rb'))

# Prepare Test Data
# from DataPreparation import normalizeImage
# testImagePath = ".//ImageProcessingPOC//00000001_002.png"
# test_data = pd.DataFrame(normalizeImage(testImagePath))
# x_test = test_data

# Predict test data result
y_pred = trained_model.predict(x_test)
print(y_pred)
