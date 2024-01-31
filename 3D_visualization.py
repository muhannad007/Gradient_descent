import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sympy import diff, symbols

def f(x, y):
    r = 3**(-x**2 - y**2)
    return 1 / (r+1)

x = np.linspace(start=-2, stop=2, num=200)
y = np.linspace(start=-2, stop=2, num=200)

x, y = np.meshgrid(x, y)

fig = plt.figure(figsize=[16, 12])
ax = fig.add_subplot(projection='3d')

ax.set_xlabel('X', fontsize=20)
ax.set_ylabel('Y', fontsize=20)
ax.set_zlabel('F(x, y)', fontsize=20)

ax.plot_surface(x, y, f(x, y), cmap=cm.coolwarm, alpha=0.4)

plt.show()
