# Data Preprocessing Tools

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('./Datasets/SampleFeaturesToDiabetesPredictionData/diabetes.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 0:8])
X[:, 0:8] = imputer.transform(X[:, 0:8])
print("missing data: ",X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
print("X_train: ",X_train)
print("X_test: ",X_test)
print("y_train: ",y_train)
print("y_test: ",y_test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, :] = sc.fit_transform(X_train[:, :])
X_test[:, :] = sc.transform(X_test[:, :])
print("Scaling X_train: ",X_train)
print("Scaling X_test: ",X_test)


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 12)
classifier.fit(X_train, y_train)
pred_y = classifier.predict(X_test)
matrix = confusion_matrix(y_test, pred_y)
print("accuracy_score: ",accuracy_score(y_test, pred_y))