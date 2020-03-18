import numpy as np

x = np.array([[1, 0], [2, 0], [3, 0], [0, 1], [0, 2]])
T = np.array([[0], [0], [1], [0], [1]])


nExamples = np.size(x, 0)
nInputs = np.size(x, 1) + 1

X = np.ones([nExamples, nInputs])
X[:, :-1] = x

W = np.array([[1], [1], [2]])

gError = 1

print("Last element ")
print(x[-1])  # Returns the last element
print("Second and third element ")
print(x[1:3])  # Returns elements 1 and 2 (3 is not included)
print("Last 2 elements ")
print(x[-1:])  # Returns elements 1 and 2 (3 is not included)
print("Subassignment of small x")
print(X)


while gError != 0:
    gError = 0
    for example in range(nExamples):
        n = np.dot(X[example, :], W)
        y = n >= 0
        e = T[example] - y
        dW = np.reshape(X[example]*e, [nInputs, 1])
        W = W + dW
        gError = gError + abs(e) != 0

print("Result of training: ")
print(W) #Wp (first row) is the weight of a punch, Wk (second row) is the weight of a kick and I don't know what's the last one


