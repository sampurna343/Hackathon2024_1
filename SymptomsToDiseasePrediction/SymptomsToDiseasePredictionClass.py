# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Importing the dataset
training_set = pd.read_csv('./Datasets/SymptomsToDiseaseData/training.csv')
train_x = training_set.iloc[:, :-1].values
train_y = training_set.iloc[:, -1].values

testing_set = pd.read_csv('./Datasets/SymptomsToDiseaseData/testing.csv')
test_x = testing_set.iloc[:, :-1].values
test_y = testing_set.iloc[:, -1].values

classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 12)
classifier.fit(train_x, train_y)

pred_y = classifier.predict(test_x)
matrix = confusion_matrix(test_y, pred_y)

print(accuracy_score(test_y, pred_y))



