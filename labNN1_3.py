from sklearn.neural_network import MLPClassifier
import numpy as np

x = np.array([[1, 0], [2, 0], [3, 0], [0, 1], [0, 2]])
T = np.array([[0], [0], [1], [0], [1]])


nExamples = np.size(x, 0)
nInputs = np.size(x, 1) + 1

X = np.ones([nExamples, nInputs])
X[:, :-1] = x

W = np.array([[1], [1], [2]])

gError = 1

net = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

net.fit(X, T)
print(net.predict(X))


#HOMEWORK: Use this stuff to solve a classification problem from uci repository
#https://archive.ics.uci.edu/ml/index.php
# Data sets -> https://archive.ics.uci.edu/ml/datasets.php
