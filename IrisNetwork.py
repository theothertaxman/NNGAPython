from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import numpy as np

iris_dataset = load_iris()

print("Target Names: {}".format(iris_dataset['target_names']))
print("Feature Names: {}".format(iris_dataset['feature_names']))
print("Type of Data: {}".format(type(iris_dataset['data'])))
print("Shape of Data: {}".format(iris_dataset['data'].shape))

print("Type of Target: {}".format(type(iris_dataset['target'])))
print("Shape of Target: {}".format(iris_dataset['target'].shape))
print("Target: \n{}".format(iris_dataset['target']))

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'],iris_dataset['target'], random_state=0)

print("X_train Shape: {}".format(X_train.shape))
print("y_train Shape: {}".format(y_train.shape))

print("X_test Shape: {}".format(X_test.shape))
print("y_test Shape: {}".format(y_test.shape))

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)


X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new Shape: {}".format(X_new.shape))

prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("Predicted target name: {}".format(iris_dataset['target_names'][prediction]))

y_pred = knn.predict(X_test)
print("Test set predictions: {}".format(y_pred))
print("Test set score(np.mean): {:.2f}".format(np.mean(y_pred == y_test)))
print("Test set score (knn.score): {:.2f}".format(knn.score(X_test, y_test)))


X_new = np.array([[6.9, 2.5, 4.7, 2.1]])
prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("Predicted target name: {}".format(iris_dataset['target_names'][prediction]))

iris_confusion_matrix = confusion_matrix(y_test, knn.predict(X_test))
print("Confusion matrix: \n {}".format(iris_confusion_matrix))
