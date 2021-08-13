import time, sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from sklearn import svm
from sklearn.datasets import make_blobs, make_moons
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, KernelDensity
from sklearn.mixture import GaussianMixture

from AD.PCAAD import PCAAD

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
        14. * (np.random.RandomState(42).rand(n_samples, 2) - 0.5)
    ]

    outliers = rng.uniform(low=-6, high=6, size=(n_outliers, 2))
    return outliers, datasets

xx, yy = np.meshgrid(np.linspace(-7, 7, 50),
                     np.linspace(-7, 7, 50))

plt.xlim(-7, 7)
plt.ylim(-7, 7)
colors = np.array(['#67ae67', '#ff7f00'])

def ocsvm():
    outliers, ds = get_random_dataset()
    print(outliers.shape)

    for dataset_i, inliers in enumerate(ds):
        X = np.concatenate([inliers, outliers], axis=0)
        print(X.shape)

        # plt.figure(figsize=(20, 10))
        plt.subplot(1, len(ds), dataset_i+1)

        oc_svm = svm.OneClassSVM(nu=outlier_percentage, kernel="rbf", gamma=0.1)
        oc_svm.fit(X)
        labels = oc_svm.predict(X)
        print(labels.shape)

        plot_space = np.c_[xx.ravel(), yy.ravel()]
        # np.set_printoptions(threshold=sys.maxsize)
        # print(plot_space)
        Z = oc_svm.decision_function(plot_space)
        Z_contours = Z.reshape(xx.shape)

        # print(Z)
        plt.contour(xx, yy, Z_contours, levels=[0], linewidths=2, colors='black')
        plt.contourf(xx, yy, Z_contours, levels=np.linspace(Z_contours.min(), Z_contours.max(), 7), cmap=plt.cm.Blues_r)
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[(labels+1) // 2])

    plt.show()

def lof():
    outliers, ds = get_random_dataset()
    print(outliers.shape)

    for dataset_i, inliers in enumerate(ds):
        X = np.concatenate([inliers, outliers], axis=0)
        print(X.shape)

        # plt.figure(figsize=(20, 10))
        plt.subplot(1, len(ds), dataset_i+1)

        lof = LocalOutlierFactor(n_neighbors=35, contamination=outlier_percentage)

        labels = lof.fit_predict(X)
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[(labels+1) // 2])

    plt.show()

def iforest():
    outliers, ds = get_random_dataset()
    print(outliers.shape)

    for dataset_i, inliers in enumerate(ds):
        X = np.concatenate([inliers, outliers], axis=0)
        print(X.shape)

        # plt.figure(figsize=(20, 10))
        plt.subplot(1, len(ds), dataset_i+1)

        iforest = IsolationForest(contamination=outlier_percentage, random_state=42)
        labels = iforest.fit_predict(X)

        plot_space = np.c_[xx.ravel(), yy.ravel()]
        # np.set_printoptions(threshold=sys.maxsize)
        # print(plot_space)
        Z = iforest.decision_function(plot_space)
        Z_contours = Z.reshape(xx.shape)

        # print(Z)
        plt.contour(xx, yy, Z_contours, levels=[0], linewidths=2, colors='black')
        plt.contourf(xx, yy, Z_contours, levels=np.linspace(Z_contours.min(), Z_contours.max(), 7), cmap=plt.cm.Blues_r)
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[(labels+1) // 2])

    plt.show()

def gaussian_mixture():
    outliers, ds = get_random_dataset()
    print(outliers.shape)

    for dataset_i, inliers in enumerate(ds):
        X = np.concatenate([inliers, outliers], axis=0)
        print(X.shape)

        # plt.figure(figsize=(20, 10))
        plt.subplot(1, len(ds), dataset_i+1)
        gmm = GaussianMixture(covariance_type='full', init_params='kmeans', max_iter=200, n_components=2)
        gmm.fit(X)

        scores = gmm.score_samples(X)
        print(scores)

        threshold = np.quantile(scores, 0.15) # 15% outliers
        labels = np.ones(X.shape[0], dtype=int)
        labels[np.where(scores < threshold)] = -1
        print(labels)

        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[(labels+1) // 2])

        plot_space = np.c_[xx.ravel(), yy.ravel()]
        # np.set_printoptions(threshold=sys.maxsize)
        # print(plot_space)
        Z = gmm.score_samples(plot_space)
        Z_contours = Z.reshape(xx.shape)

        # print(Z)
        contours = plt.contour(xx, yy, Z_contours, linewidths=2)
        # plt.colorbar(contours, shrink=0.8, extend='both')

    plt.show()

def kde():
    outliers, ds = get_random_dataset()
    print(outliers.shape)

    for dataset_i, inliers in enumerate(ds):
        X = np.concatenate([inliers, outliers], axis=0)
        print(X.shape)

        # plt.figure(figsize=(20, 10))
        plt.subplot(1, len(ds), dataset_i+1)
        kde = KernelDensity(algorithm='auto', bandwidth=1.0, kernel='gaussian', leaf_size=40, metric='euclidean')
        kde.fit(X)

        scores = kde.score_samples(X)
        print(scores)

        threshold = np.quantile(scores, 0.15) # 15% outliers
        labels = np.ones(X.shape[0], dtype=int)
        labels[np.where(scores < threshold)] = -1
        print(labels)

        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[(labels+1) // 2])

        plot_space = np.c_[xx.ravel(), yy.ravel()]
        # np.set_printoptions(threshold=sys.maxsize)
        # print(plot_space)
        Z = kde.score_samples(plot_space)
        Z_contours = Z.reshape(xx.shape)

        # print(Z)
        contours = plt.contour(xx, yy, Z_contours, linewidths=2)
        # plt.colorbar(contours, shrink=0.8, extend='both')

    plt.show()

def pcaad():
    outliers, ds = get_random_dataset()
    print(outliers.shape)

    for dataset_i, inliers in enumerate(ds):
        X = np.concatenate([inliers, outliers], axis=0)
        print(X.shape)

        # plt.figure(figsize=(20, 10))
        plt.subplot(1, len(ds), dataset_i+1)
        pcaad = PCAAD()
        pcaad.fit(X)
        labels = pcaad.predict(X)

        print(labels.shape)

        plot_space = np.c_[xx.ravel(), yy.ravel()]
        # np.set_printoptions(threshold=sys.maxsize)
        # print(plot_space)
        Z = pcaad.predict(plot_space)
        # Z_contours = Z.reshape(xx.shape)

        # print(Z)
        # plt.contour(xx, yy, Z_contours, levels=[0], linewidths=2, colors='black')
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[(labels+1) // 2])

    plt.show()
