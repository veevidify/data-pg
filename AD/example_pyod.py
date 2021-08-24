import time, sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from sklearn.datasets import make_blobs, make_moons

from pyod.models.pca import PCA
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.loda import LODA
from pyod.models.knn import KNN
from pyod.models.abod import ABOD

from sklearn.mixture import GaussianMixture
from pyod.utils.utility import standardizer
from pyod.models.combination import average as ensemble_average
from pyod.models.combination import majority_vote as ensemble_instances

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

xx, yy = np.meshgrid(np.linspace(-7, 7, 75),
                     np.linspace(-7, 7, 75))

plt.xlim(-7, 7)
plt.ylim(-7, 7)
colors = np.array(['#ff7f00','#67ae98'])

def ocsvm():
    outliers, ds = get_random_dataset()
    print(outliers.shape)

    for dataset_i, inliers in enumerate(ds):
        X = np.concatenate([inliers, outliers], axis=0)
        print(X.shape)

        # plt.figure(figsize=(20, 10))
        plt.subplot(1, len(ds), dataset_i+1)

        oc_svm = OCSVM(contamination=outlier_percentage, kernel="rbf", gamma='auto')
        oc_svm.fit(X)
        labels = oc_svm.predict(X)
        print(labels.shape)

        plot_space = np.c_[xx.ravel(), yy.ravel()]
        # np.set_printoptions(threshold=sys.maxsize)
        # print(plot_space)

        Z = oc_svm.predict(plot_space)
        print(Z)
        Z_contours = Z.reshape(xx.shape)
        plt.contour(xx, yy, Z_contours, levels=[0.5], linewidths=2, colors='black')

        Z = -1.0 * oc_svm.decision_function(plot_space)
        Z_contours = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z_contours, levels=np.linspace(Z_contours.min(), Z_contours.max(), 8), cmap=plt.cm.Blues_r)

        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[labels])

    plt.show()

def lof():
    outliers, ds = get_random_dataset()
    print(outliers.shape)

    for dataset_i, inliers in enumerate(ds):
        X = np.concatenate([inliers, outliers], axis=0)
        print(X.shape)

        # plt.figure(figsize=(20, 10))
        plt.subplot(1, len(ds), dataset_i+1)

        lof = LOF(n_neighbors=35, leaf_size=40, algorithm='auto', metric='minkowski', contamination=outlier_percentage)

        lof.fit(X)
        labels = lof.predict(X)
        print(labels.shape)

        plot_space = np.c_[xx.ravel(), yy.ravel()]
        # np.set_printoptions(threshold=sys.maxsize)
        # print(plot_space)

        Z = lof.predict(plot_space)
        print(Z)
        Z_contours = Z.reshape(xx.shape)
        plt.contour(xx, yy, Z_contours, levels=[0.5], linewidths=2, colors='black')

        Z = -1.0 * lof.decision_function(plot_space)
        Z_contours = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z_contours, levels=np.linspace(Z_contours.min(), Z_contours.max(), 8), cmap=plt.cm.Blues_r)

        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[labels])

    plt.show()

def iforest():
    outliers, ds = get_random_dataset()
    print(outliers.shape)

    for dataset_i, inliers in enumerate(ds):
        X = np.concatenate([inliers, outliers], axis=0)
        print(X.shape)

        # plt.figure(figsize=(20, 10))
        plt.subplot(1, len(ds), dataset_i+1)

        iforest = IForest(n_estimators=400, max_samples='auto', contamination=outlier_percentage)

        iforest.fit(X)
        labels = iforest.predict(X)
        print(labels.shape)

        plot_space = np.c_[xx.ravel(), yy.ravel()]
        # np.set_printoptions(threshold=sys.maxsize)
        # print(plot_space)

        Z = iforest.predict(plot_space)
        print(Z)
        Z_contours = Z.reshape(xx.shape)
        plt.contour(xx, yy, Z_contours, levels=[0.5], linewidths=2, colors='black')

        Z = -1.0 * iforest.decision_function(plot_space)
        Z_contours = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z_contours, levels=np.linspace(Z_contours.min(), Z_contours.max(), 8), cmap=plt.cm.Blues_r)

        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[labels])

    plt.show()


def pcaad():
    outliers, ds = get_random_dataset()
    print(outliers.shape)

    for dataset_i, inliers in enumerate(ds):
        X = np.concatenate([inliers, outliers], axis=0)
        print(X.shape)

        # plt.figure(figsize=(20, 10))
        plt.subplot(1, len(ds), dataset_i+1)

        pcaad = PCA(contamination=outlier_percentage)

        pcaad.fit(X)
        labels = pcaad.predict(X)
        print(labels.shape)

        plot_space = np.c_[xx.ravel(), yy.ravel()]
        # np.set_printoptions(threshold=sys.maxsize)
        # print(plot_space)

        Z = pcaad.predict(plot_space)
        print(Z)
        Z_contours = Z.reshape(xx.shape)
        plt.contour(xx, yy, Z_contours, levels=[0.5], linewidths=2, colors='black')

        Z = -1.0 * pcaad.decision_function(plot_space)
        Z_contours = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z_contours, levels=np.linspace(Z_contours.min(), Z_contours.max(), 8), cmap=plt.cm.Blues_r)

        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[labels])

    plt.show()

def loda():
    outliers, ds = get_random_dataset()
    print(outliers.shape)

    for dataset_i, inliers in enumerate(ds):
        X = np.concatenate([inliers, outliers], axis=0)
        print(X.shape)

        # plt.figure(figsize=(20, 10))
        plt.subplot(1, len(ds), dataset_i+1)

        loda = LODA(contamination=outlier_percentage, n_bins=20, n_random_cuts=400)

        loda.fit(X)
        labels = loda.predict(X)
        print(labels.shape)

        plot_space = np.c_[xx.ravel(), yy.ravel()]
        # np.set_printoptions(threshold=sys.maxsize)
        # print(plot_space)

        Z = loda.predict(plot_space)
        print(Z)
        Z_contours = Z.reshape(xx.shape)
        plt.contour(xx, yy, Z_contours, levels=[0.5], linewidths=2, colors='black')

        Z = -1.0 * loda.decision_function(plot_space)
        Z_contours = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z_contours, levels=np.linspace(Z_contours.min(), Z_contours.max(), 10), cmap=plt.cm.Blues_r)

        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[labels])

    plt.show()

def knn():
    outliers, ds = get_random_dataset()
    print(outliers.shape)

    for dataset_i, inliers in enumerate(ds):
        X = np.concatenate([inliers, outliers], axis=0)
        print(X.shape)

        # plt.figure(figsize=(20, 10))
        plt.subplot(1, len(ds), dataset_i+1)

        knn = KNN(contamination=outlier_percentage, n_neighbors=35, method='largest', leaf_size=40, radius=0.1, metric='minkowski', metric_params=None)

        knn.fit(X)
        labels = knn.predict(X)
        print(labels.shape)

        plot_space = np.c_[xx.ravel(), yy.ravel()]
        # np.set_printoptions(threshold=sys.maxsize)
        # print(plot_space)

        Z = knn.predict(plot_space)
        print(Z)
        Z_contours = Z.reshape(xx.shape)
        plt.contour(xx, yy, Z_contours, levels=[0.5], linewidths=2, colors='black')

        Z = -1.0 * knn.decision_function(plot_space)
        Z_contours = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z_contours, levels=np.linspace(Z_contours.min(), Z_contours.max(), 10), cmap=plt.cm.Blues_r)

        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[labels])

    plt.show()

def fastabod():
    outliers, ds = get_random_dataset()
    print(outliers.shape)

    for dataset_i, inliers in enumerate(ds):
        X = np.concatenate([inliers, outliers], axis=0)
        print(X.shape)

        # plt.figure(figsize=(20, 10))
        plt.subplot(1, len(ds), dataset_i+1)

        fastabod = ABOD(contamination=outlier_percentage, n_neighbors=35, method='fast')

        fastabod.fit(X)
        labels = fastabod.predict(X)
        print(labels.shape)

        plot_space = np.c_[xx.ravel(), yy.ravel()]
        # np.set_printoptions(threshold=sys.maxsize)
        # print(plot_space)

        # Z = fastabod.predict(plot_space)
        # print(Z)
        # Z_contours = Z.reshape(xx.shape)
        # plt.contour(xx, yy, Z_contours, levels=[0.5], linewidths=2, colors='black')

        # Z = -1.0 * fastabod.decision_function(plot_space)
        # Z_contours = Z.reshape(xx.shape)
        # plt.contourf(xx, yy, Z_contours, levels=np.linspace(Z_contours.min(), Z_contours.max(), 10), cmap=plt.cm.Blues_r)

        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[labels])

    plt.show()

def ensemble_knn():
    outliers, ds = get_random_dataset()
    print(outliers.shape)

    for dataset_i, inliers in enumerate(ds):
        X = np.concatenate([inliers, outliers], axis=0)
        print(X.shape)

        # plt.figure(figsize=(20, 10))
        plt.subplot(1, len(ds), dataset_i+1)

        nns = [5, 25, 35, 45, 55]
        rs = [0.1, 0.4, 0.8, 1.0, 1.2]
        ms = ['euclidean', 'minkowski']
        scores = np.zeros((X.shape[0], len(nns) * len(rs) * len(ms)))
        i = 0
        # TODO: use itertools
        for nn in nns:
            for r in rs:
                for m in ms:
                    knn_i = KNN(contamination=outlier_percentage, n_neighbors=nn, method='largest', leaf_size=nn, radius=r, metric=m, metric_params=None)
                    knn_i.fit(X)
                    scores[:, i] = knn_i.decision_scores_
                    i = i + 1

        std_scores = standardizer(scores)
        combi_avg = ensemble_average(std_scores)

        threshold = np.quantile(combi_avg, 0.85) # 15% outliers
        labels = np.zeros(X.shape[0], dtype=int)
        labels[np.where(combi_avg > threshold)] = 1
        print(labels)

        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[labels])

    plt.show()