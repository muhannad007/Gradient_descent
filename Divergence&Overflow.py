import numpy as np
import matplotlib.pyplot as plt
from gradient import gradient_descent

x_1 = np.linspace(start=-2.5, stop=2.5, num=1000)

def f(x):
    return x**5 - 2*x**4 + 2
def df(x):
    return 5*x**4 - 8*x**3

new_x, list_x, deriv_list = gradient_descent(derivative_func=df, initial_guess=-0.2, max_iter=70)

plt.figure(figsize=[15, 5])

plt.subplot(1, 2, 1)
plt.xlim([-1.5, 2.5])
plt.ylim([-1, 4])
plt.title('Cost Fuction', fontsize = 17)
plt.xlabel('x', fontsize = 16)
plt.ylabel('f(x)', fontsize = 16)
plt.plot(x_1, f(x_1), color = 'blue', linewidth = 3)
# values = np.array(x_list)
plt.scatter(list_x, f(np.array(list_x)), color = 'red', s = 100, alpha = 0.6)

plt.subplot(1, 2, 2)
plt.grid()
plt.title('Slope of Cost Fuction', fontsize = 17)
plt.xlabel('x', fontsize = 16)
plt.ylabel('df(x)', fontsize = 16)
plt.xlim(-1, 2)
plt.ylim(-4, 5)
plt.plot(x_1, df(x_1), color = 'skyblue', linewidth = 5, alpha = 0.6)
plt.scatter(list_x, deriv_list, color = 'red', s = 100, alpha = 0.5)

plt.show()

