import numpy as np
import matplotlib.pyplot as plt
# from sympy import symbols, diff
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import math

def f(x, y):
    r = 3**(-x**2 - y**2)
    return 1 / (r+1)

def fpx(x, y):
    r = 3**(-x**2 - y**2)
    return 2*x*math.log(3)*r / (r+1)**2

def fpy(x, y):
    r = 3**(-x**2 - y**2)
    return 2*y*math.log(3)*r / (r+1)**2

x = np.linspace(start=-2, stop=2, num=200)
y = np.linspace(start=-2, stop=2, num=200)

x, y = np.meshgrid(x, y)

# a, b = symbols('x, y')

multiplier = 0.1
max_iter = 500
params = np.array([1.8, 1.0]) #instead of new_x in the 2D dimensions
values = params.reshape(1, 2)

for n in range(max_iter):
    gradient_x = fpx(params[0], params[1]) # partial derivative of f at params point
    gradient_y = fpy(params[0], params[1]) # partial derivative af f at params point
    # gradient_x = diff(f(a, b), a).evalf(subs = {a: params[0], b: params[1]})
    # gradient_y = diff(f(a, b), b).evalf(subs = {a: params[0], b: params[1]})
    gradient = np.array([gradient_x, gradient_y])
    params = params - multiplier * gradient # updating the params values
    values = np.append(values, params.reshape(1, 2), axis=0)

fig = plt.figure(figsize=[16, 12])
ax = fig.add_subplot(projection='3d')

ax.set_xlabel('X', fontsize=20)
ax.set_ylabel('Y', fontsize=20)
ax.set_zlabel('f(x, y)', fontsize=20)

ax.plot_surface(x, y, f(x, y), cmap=cm.coolwarm, alpha=0.4)
ax.scatter(values[:, 0], values[:, 1], f(values[:, 0], values[:, 1]), s=50, color='red')

plt.show()


# print (gradient)
# print (params[0])
# print (params[1])
# print (f(params[0], params[1]))
