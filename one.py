import numpy as np



a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

c = np.concatenate((a, b), axis = 0)
c = np.concatenate((c, np.flip(c, axis = 0)),  axis= 0)

print(c)
print(c.shape)


