import time, sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import svm
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, KernelDensity
from sklearn.mixture import GaussianMixture

import AD.utils as utils

from decomposition.PCA import PCA, PCA2
from AD.PCAAD import PCAAD

n_samples = 300
outlier_percentage = 0.15

n_outliers = int(n_samples * outlier_percentage)
n_inliers = n_samples - n_outliers

rng = np.random.RandomState(42)
colors = np.array(['#377eb8', '#ff7f00'])

def get_random_dataset():
    blobs_params = dict(random_state=0, n_samples=n_inliers, n_features=10)
    datasets = [
        make_blobs(
            centers=[np.zeros(10), np.zeros(10)],
            cluster_std=0.5,
            **blobs_params)[0],
        make_blobs(
            centers=[2*np.ones(10), (-2)*np.ones(10)],
            cluster_std=[0.5, 0.5],
            **blobs_params)[0],
        make_blobs(
            centers=[2*np.ones(10), (-2)*np.ones(10)],
            cluster_std=[1.5, .3],
            **blobs_params)[0],
        # 14. * (np.random.RandomState(42).rand(n_samples, 10) - 0.5)
    ]

    outliers = rng.uniform(low=-6, high=6, size=(n_outliers, 10))
    return outliers, datasets

def ocsvm():
    outliers, ds = get_random_dataset()

    for dataset_i, inliers in enumerate(ds):
        X = np.concatenate([inliers, outliers], axis=0)
        # X = inliers
        print(X.shape)

        X_scaled = StandardScaler().fit_transform(X)
        # X_scaled = X

        oc_svm = svm.OneClassSVM(nu=outlier_percentage, kernel="rbf", gamma=0.05)
        oc_svm.fit(X_scaled)
        labels = oc_svm.predict(X_scaled)
        print(labels.shape)

        pca = utils.get_pca(X)
        pca["Labels"] = labels
        print(pca.head())

        plx = pca['PC1']
        ply = pca['PC2']
        plz = pca['PC3']

        # 2d scatterplot
        # plt.figure(figsize=(20, 10))
        # sns.scatterplot(x=pca["PC1"], y=np.zeros(y.shape[0]), hue=pca["Labels"], s=200)
        # sns.scatterplot(x=pca["PC1"], y=pca["PC2"], hue=pca["Labels"], s=200)

        # 3d scatterplot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.scatter(plx, ply, plz, color=colors[(labels+1)//2]) # colors -1, 1 mapped to 0, 1

    plt.show()

def lof():
    outliers, ds = get_random_dataset()

    for dataset_i, inliers in enumerate(ds):
        X = np.concatenate([inliers, outliers], axis=0)
        # X = inliers
        print(X.shape)

        X_scaled = StandardScaler().fit_transform(X)
        # X_scaled = X

        lof = LocalOutlierFactor(n_neighbors=40, contamination=outlier_percentage)

        labels = lof.fit_predict(X)

        # 2D plot

        pca = utils.get_pca(X)
        pca["Labels"] = labels
        print(pca.head())

        plx = pca['PC1']
        ply = pca['PC2']
        plz = pca['PC3']

        # 2d scatterplot
        # plt.figure(figsize=(20, 10))
        # sns.scatterplot(x=pca["PC1"], y=np.zeros(y.shape[0]), hue=pca["Labels"], s=200)
        # sns.scatterplot(x=pca["PC1"], y=pca["PC2"], hue=pca["Labels"], s=200)

        # 3d
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.scatter(plx, ply, plz, color=colors[(labels+1)//2]) # colors -1, 1 mapped to 0, 1

    plt.show()

def iforest():
    outliers, ds = get_random_dataset()

    for dataset_i, inliers in enumerate(ds):
        X = np.concatenate([inliers, outliers], axis=0)
        # X = inliers
        print(X.shape)

        X_scaled = StandardScaler().fit_transform(X)
        # X_scaled = X

        iforest = IsolationForest(contamination=outlier_percentage, random_state=42)
        labels = iforest.fit_predict(X)

        # 2D plot
        pca = utils.get_pca(X)
        pca["Labels"] = labels
        print(pca.head())

        plx = pca['PC1']
        ply = pca['PC2']
        plz = pca['PC3']

        # 2d scatterplot
        # plt.figure(figsize=(20, 10))
        # sns.scatterplot(x=pca["PC1"], y=np.zeros(y.shape[0]), hue=pca["Labels"], s=200)
        # sns.scatterplot(x=pca["PC1"], y=pca["PC2"], hue=pca["Labels"], s=200)

        # 3d
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.scatter(plx, ply, plz, color=colors[(labels+1)//2]) # colors -1, 1 mapped to 0, 1

    plt.show()

def gaussian_mixture():
    outliers, ds = get_random_dataset()

    for dataset_i, inliers in enumerate(ds):
        X = np.concatenate([inliers, outliers], axis=0)
        # X = inliers
        print(X.shape)

        X_scaled = StandardScaler().fit_transform(X)
        # X_scaled = X

        gmm = GaussianMixture(covariance_type='full', init_params='kmeans', max_iter=200, n_components=2)
        gmm.fit(X)

        scores = gmm.score_samples(X)
        threshold = np.quantile(scores, 0.15) # 15% outliers
        labels = np.ones(X.shape[0], dtype=int)
        labels[np.where(scores < threshold)] = -1
        print(labels)

        # 2D plot
        pca = utils.get_pca(X)
        pca["Labels"] = labels
        print(pca.head())

        plx = pca['PC1']
        ply = pca['PC2']
        plz = pca['PC3']

        # 2d scatterplot
        # plt.figure(figsize=(20, 10))
        # sns.scatterplot(x=pca["PC1"], y=np.zeros(y.shape[0]), hue=pca["Labels"], s=200)
        # sns.scatterplot(x=pca["PC1"], y=pca["PC2"], hue=pca["Labels"], s=200)

        # 3d
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.scatter(plx, ply, plz, color=colors[(labels+1)//2]) # colors -1, 1 mapped to 0, 1
        # ax.scatter(plx, ply, plz, c=scores)

    plt.show()

def kde():
    outliers, ds = get_random_dataset()

    for dataset_i, inliers in enumerate(ds):
        X = np.concatenate([inliers, outliers], axis=0)
        # X = inliers
        print(X.shape)

        X_scaled = StandardScaler().fit_transform(X)
        # X_scaled = X

        kde = KernelDensity(algorithm='auto', bandwidth=0.5, kernel='gaussian', leaf_size=40, metric='minkowski')
        kde.fit(X)

        scores = kde.score_samples(X)
        threshold = np.quantile(scores, 0.15) # 15% outliers
        labels = np.ones(X.shape[0], dtype=int)
        labels[np.where(scores < threshold)] = -1
        print(labels)

        # 2D plot
        pca = utils.get_pca(X)
        pca["Labels"] = labels
        print(pca.head())

        plx = pca['PC1']
        ply = pca['PC2']
        plz = pca['PC3']

        # 2d scatterplot
        # plt.figure(figsize=(20, 10))
        # sns.scatterplot(x=pca["PC1"], y=np.zeros(y.shape[0]), hue=pca["Labels"], s=200)
        # sns.scatterplot(x=pca["PC1"], y=pca["PC2"], hue=pca["Labels"], s=200)

        # 3d
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.scatter(plx, ply, plz, color=colors[(labels+1)//2]) # colors -1, 1 mapped to 0, 1
        # ax.scatter(plx, ply, plz, c=scores)

    plt.show()

def pcaad():
    outliers, ds = get_random_dataset()

    for dataset_i, inliers in enumerate(ds):
        X = np.concatenate([inliers, outliers], axis=0)
        # X = inliers
        print(X.shape)

        X_scaled = StandardScaler().fit_transform(X)
        # X_scaled = X

        pcaad = PCAAD()
        pcaad.fit(X)
        labels = pcaad.predict(X)

        # 2D plot
        pca = utils.get_pca(X)
        pca["Labels"] = labels
        print(pca.head())

        plx = pca['PC1']
        ply = pca['PC2']
        plz = pca['PC3']

        # 2d scatterplot
        # plt.figure(figsize=(20, 10))
        # sns.scatterplot(x=pca["PC1"], y=np.zeros(y.shape[0]), hue=pca["Labels"], s=200)
        # sns.scatterplot(x=pca["PC1"], y=pca["PC2"], hue=pca["Labels"], s=200)

        # 3d
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.scatter(plx, ply, plz, color=colors[(labels+1)//2]) # colors -1, 1 mapped to 0, 1

    plt.show()
