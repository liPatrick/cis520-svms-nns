import numpy as np 

xs = np.array([(1),(2),(3)]).reshape((3,1))
print(xs)
ones = np.ones((1,3))
print(ones)
print("dot product")
print(np.matmul(xs.T, ones.T))


print(xs.mean())

print(xs.shape)
weights = np.array([(1,2,3), (1,2,3), (1,2,3)])
print("weights, ", weights.mean(axis=1))

tiled = np.tile(xs, (4,1))
print(np.transpose(tiled))


def sigmoid(g):
    return 1./(1. + np.exp(-g))

vectorizedg = np.vectorize(sigmoid)
print(vectorizedg(tiled))


#product = np.einsum("i1,j1->j1", ibyone, onebyj)
#print(product)