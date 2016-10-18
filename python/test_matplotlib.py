#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(-np.pi, np.pi, 100, endpoint=True)
C, S = np.cos(X), np.sin(X)

index_list = [np.random.randint(100) for i in range(10)]
print(index_list)
Px = [X[i] for i in index_list]
Py = [C[i] for i in index_list]

plt.plot(X, C)
plt.plot(X, S)
plt.scatter(Px, Py)


plt.show()
# plt.pause(0.1)
