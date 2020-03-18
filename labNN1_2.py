import numpy as np

x = np.array([[1, 0], [2, 0], [3, 0], [0, 1], [0, 2]])
T = np.array([[0], [0], [1], [0], [1]])


nExamples = np.size(x, 0)
nInputs = np.size(x, 1) + 1

X = np.ones([nExamples, nInputs])
X[:, :-1] = x

W = np.array([[1], [1], [2]])
error = 1


def myOutput(W, X):
    N = np.dot(X,W)
    return(N >= 0)


while error != 0:
    Y = myOutput(W, X)
    E = T - Y   #Error is Target - Y (The output of the network)
    dW = X * E
    print("Initial dW: ")
    print(dW)
    print('===============')
    dW = np.sum(dW, axis=0) #Sum of each column in dW matrix
    dW = np.reshape(dW, [nInputs, 1])
    W = W + dW
    error = sum(abs(E))
    print("Sum of matrix")
    print(dW)
    print('===============')
    #print(W)

print(myOutput(W, X))
