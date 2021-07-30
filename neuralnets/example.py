import numpy as np
from .models import linear_regression
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

fig = plt.figure()
ax = plt.axes(projection='3d')

def show():
    X = np.random.rand(100, 2)
    y = 2*X[:,0] + 2 * X[:, 1] + 4 + .2*np.random.randn(100)

    model = linear_regression(X, y)
    weights = model.get_weights()
    print(weights)

    ax.scatter3D(X[:,0], X[:,1], y, c=y, cmap='gray')

    X1, X2 = np.meshgrid(np.arange(0.0, 1.0, 0.1), np.arange(0.0, 1.0, 0.1))
    model_Y = weights[0][0]*X1 + weights[0][1]*X2 + weights[1]

    ax.plot_surface(X1, X2, model_Y)
    plt.show()
