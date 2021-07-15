import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from matplotlib.backends.backend_pdf import PdfPages

np.random.seed(21)
def show():
    means = [[2, 2], [4, 1]]
    cov = [[.3, .2], [.2, .3]]
    N = 10
    X0 = np.random.multivariate_normal(means[0], cov, N)
    X1 = np.random.multivariate_normal(means[1], cov, N)
    X1[-1, :] = [2.7, 2]
    X = np.concatenate((X0.T, X1.T), axis = 1)
    y = np.concatenate((np.ones((1, N)), -1*np.ones((1, N))), axis = 1)



