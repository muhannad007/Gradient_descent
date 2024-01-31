import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**4 - 4*x**2 + 5
def df(x):
    return 4*x**3 - 8*x

x_1 = np.linspace(start=-2, stop=2, num=1000)

def gradient_descent(derivative_func, initial_guess, multiplier=0.02, precision=0.001, max_iter=300):
    new_x = initial_guess

    x_list = [new_x]
    slope_list = [derivative_func(new_x)]

    for n in range(max_iter):
        prev_x = new_x
        gradient = derivative_func(prev_x)
        new_x = prev_x - multiplier * gradient
        step_size = abs(new_x - prev_x)
        x_list.append(new_x)
        slope_list.append(derivative_func(new_x))

        if step_size < precision:
            break
    return new_x, x_list, slope_list

local_min, list_x, deriv_list = gradient_descent(derivative_func=df, initial_guess=0.1)



# print('The minimun occurs at this point; ', new_x)
# print('Slope at this point: ', df(new_x))
# print('Cost at this point: ', f(new_x))

#plot the cost fucntion and derivative
plt.figure(figsize=[15, 5])

plt.subplot(1, 2, 1)
plt.xlim([-2, 2])
plt.ylim([0.5, 5.5])
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
plt.xlim(-2, 2)
plt.ylim(-6, 8)
plt.plot(x_1, df(x_1), color = 'skyblue', linewidth = 5, alpha = 0.6)
plt.scatter(list_x, deriv_list, color = 'red', s = 100, alpha = 0.5)

plt.show()
