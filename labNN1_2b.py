# x=[[[1,0],[2,0],[3,0],[0,1],[0,2]],[[1,0],[2,0],[3,0],[0,1],[0,2]]]
import numpy as np

x = [[1, 0], [2, 0], [3, 0], [0, 1], [0, 2]]
nExamples = np.size(x, 0)
nInputs = np.size(x, 1) + 1

T = np.array([[0], [0], [1], [0], [1]])


def myOutput(W, X):
    N = np.dot(X, W)
    return (N >= 0)


X = np.ones([nExamples, nInputs])
X[:, :-1] = np.array(x)

# W=np.random.randint(10,size=(nInputs,1))
W = np.array([[1], [1], [2]])

error = 1

while error != 0:
    Y = myOutput(W, X)
    E = T - Y
    dW = X * E
    dW = np.sum(dW, axis=0)
    dW = np.reshape(dW, [nInputs, 1])
    W = W + dW
    error = sum(abs(E))
    print(W)
    print('===============')

print(myOutput(W, X))