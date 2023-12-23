from random import random
import numpy as np
import matplotlib.pyplot as plt

U1 = np.random.rand(1, 100)[0]
U2 = np.random.rand(1, 100)[0]

# print(U1)
# print(U2)

N1 = (-2 * np.log(U1)) ** 0.5 * np.cos(2 * np.pi * U2)
N2 = (-2 * np.log(U1)) ** 0.5 * np.sin(2 * np.pi * U2)

plt.plot([i / 100 for i in range(0, 100)], N1)
plt.plot([i / 100 for i in range(0, 100)], N2)
plt.show()
