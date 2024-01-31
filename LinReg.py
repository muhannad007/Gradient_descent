from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

x = np.array([[0.1, 1.2, 2.4, 3.2, 4.1, 5.6, 7.2]]).transpose()
y = np.array([[1.7, 2.4, 3.5, 3.0, 6.1, 9.4, 8.2]]).transpose()

# regr = LinearRegression()
# regr.fit(x, y)

# y_hat = 1.0738368910782703 + 1.1253420908593321*x

def MSE(y, y_hat):
    m = (1/y.size) * sum((y-y_hat)**2)
    return m

def grad(x, y, thetas):
    n = y.size

    theta0_slope = (-1/2) * sum(y - thetas[0] - thetas[1]*x)
    theta1_slope = (-1/2) * sum((y - thetas[0] - thetas[1]*x)*x)

    return np.array([theta0_slope[0], theta1_slope[0]])

multiplier = 0.01
thetas = np.array([2.9, 2.9])

# collecting points for the scatter plot
plot_vals = thetas.reshape(1, 2)
mse_vals = MSE(y, thetas[0] + thetas[1]*x)

for i in range(1000):
    thetas = thetas - multiplier * grad(x, y, thetas)
    # Append the new values to the numpy array
    plot_vals = np.concatenate((plot_vals, thetas.reshape(1, 2)), axis=0)
    mse_vals = np.append(arr=mse_vals, values=MSE(y, thetas[0] + thetas[1]*x))




# print (thetas[0])
# print (thetas[1])
# print (MSE(y, thetas[1] + thetas[1]*x))

num_thetas = 200
th_0 = np.linspace(start=-1, stop=3, num=num_thetas)
th_1 = np.linspace(start=-1, stop=3, num=num_thetas)
plot_0, plot_1 = np.meshgrid(th_0, th_1)

plot_cost = np.zeros((num_thetas, num_thetas))

for i in range(num_thetas):
    for j in range(num_thetas):
        y_hat = plot_0[i][j] + plot_1[i][j]*x
        plot_cost[i][j] = MSE(y, y_hat)

fig = plt.figure(figsize=[16, 12])
ax = fig.add_subplot(projection='3d')

ax.set_title('Mean Square Error Function with Gradient Descent')
ax.set_xlabel('theta_0', fontsize=20)
ax.set_ylabel('theta_1', fontsize=20)
ax.set_zlabel('cost - MSE', fontsize=20)

ax.scatter(plot_vals[:, 0], plot_vals[:, 1], mse_vals, s=80, color='black')
ax.plot_surface(plot_0, plot_1, plot_cost, cmap=cm.rainbow, alpha=0.6)
plt.show()



# print(MSE(y, y_hat))
# plt.title('Linear Regression')
# plt.scatter(x, y, s=50, color='green')
# plt.plot(x, regr.predict(x), color='blue', linewidth=3)
# plt.xlabel('x values')
# plt.ylabel('y values')
# plt.show()

# print (regr.intercept_[0])
# print (regr.coef_[0][0])
