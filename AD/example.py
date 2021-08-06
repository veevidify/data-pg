import time, sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.datasets import make_blobs, make_moons
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

n_samples = 300
outlier_percentage = 0.15

n_outliers = int(n_samples * outlier_percentage)
n_inliers = n_samples - n_outliers

rng = np.random.RandomState(42)

def get_random_dataset():
    blobs_params = dict(random_state=0, n_samples=n_inliers, n_features=2)
    datasets = [
        make_blobs(centers=[[0, 0], [0, 0]], cluster_std=0.5,
                **blobs_params)[0],
        make_blobs(centers=[[2, 2], [-2, -2]], cluster_std=[0.5, 0.5],
                **blobs_params)[0],
        make_blobs(centers=[[2, 2], [-2, -2]], cluster_std=[1.5, .3],
                **blobs_params)[0],
        4. * (make_moons(n_samples=n_samples, noise=.05, random_state=0)[0] -
            np.array([0.5, 0.25])),
        14. * (np.random.RandomState(42).rand(n_samples, 2) - 0.5)]

    outliers = rng.uniform(low=-6, high=6, size=(n_outliers, 2))
    return outliers, datasets

xx, yy = np.meshgrid(np.linspace(-7, 7, 50),
                     np.linspace(-7, 7, 50))

plt.xlim(-7, 7)
plt.ylim(-7, 7)
colors = np.array(['#377eb8', '#ff7f00'])

def show():
    outliers, ds = get_random_dataset()
    print(outliers.shape)

    for dataset_i, inliers in enumerate(ds):
        X = np.concatenate([inliers, outliers], axis=0)
        print(X.shape)

        plt.subplot(1, len(ds), dataset_i+1)

        oc_svm = svm.OneClassSVM(nu=outlier_percentage, kernel="rbf", gamma=0.1)
        oc_svm.fit(X)
        labels = oc_svm.predict(X)

        plot_space = np.c_[xx.ravel(), yy.ravel()]
        # np.set_printoptions(threshold=sys.maxsize)
        # print(plot_space)
        Z = oc_svm.predict(plot_space)
        Z_contours = Z.reshape(xx.shape)

        # print(Z)
        plt.contour(xx, yy, Z_contours, levels=[0], linewidths=2, colors='black')

        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[(labels+1) // 2])
        # plt.scatter(xx, yy, s=10, color=colors[(Z+1) // 2])


    plt.show()

