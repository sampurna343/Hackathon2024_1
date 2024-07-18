from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
import matplotlib.pyplot as plt

d_set = load_digits()
data = d_set.data
target = d_set.target
x_train,x_test,y_train,y_test = train_test_split(data,target,test_size=0.3,random_state=42)
d_tree = DecisionTreeClassifier()
d_tree.fit(x_train,y_train)
y_pred = d_tree.predict(x_test)

result = confusion_matrix(y_test,y_pred)
acc = accuracy_score(y_test,y_pred)

print ("Confusion Matrix: ",result)
print ("Accuracy Score: ",acc)

a = int(input('type a index of the dataset which you want to check: '))
pre = d_tree.predict([x_test[a]])
print ("predicted label: ",pre)
print ("Actual label: ",y_test[a])

#we need the 2d image structure for the imshow() method
trans  = []
y = 0
for x in range(8):
	trans.append(x_test[a][y:y+8])
	y = y+8

plt.figure('that figure', figsize=(3,3))
plt.imshow(trans, cmap=plt.cm.gray_r, interpolation='nearest') #the .images[] actually stacks the linear vector into 2d image format
plt.show()