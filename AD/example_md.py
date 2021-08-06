import time, sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import svm
from sklearn.datasets import make_blobs, make_moons
# from sklearn.preprocessing import StandardScaler

from decomposition.PCA import PCA, PCA2

n_samples = 300
outlier_percentage = 0.15

n_outliers = int(n_samples * outlier_percentage)
n_inliers = n_samples - n_outliers

rng = np.random.RandomState(42)


def get_random_dataset():
    blobs_params = dict(random_state=0, n_samples=n_inliers, n_features=10)
    datasets = [
        make_blobs(
            centers=[np.zeros(10), np.zeros(10)],
            cluster_std=0.5,
            **blobs_params)[0],
        # make_blobs(
        #     centers=[[2, 2], [-2, -2]],
        #     cluster_std=[0.5, 0.5],
        #     **blobs_params)[0],
        # make_blobs(
        #     centers=[[2, 2], [-2, -2]],
        #     cluster_std=[1.5, .3],
        #     **blobs_params)[0],
        # 14. * (np.random.RandomState(42).rand(n_samples, 10) - 0.5)
    ]

    outliers = rng.uniform(low=-6, high=6, size=(n_outliers, 10))
    return outliers, datasets

def show():
    outliers, ds = get_random_dataset()

    for dataset_i, inliers in enumerate(ds):
        X = np.concatenate([inliers, outliers], axis=0)
        print(X.shape)

        oc_svm = svm.OneClassSVM(nu=outlier_percentage, kernel="rbf", gamma=0.1)
        oc_svm.fit(X)
        labels = oc_svm.predict(X)
        print(labels.shape)

        # X_scaled = StandardScaler().fit_transform(X)
        X_scaled = X
        X_mc,explained,U,S,Vt = PCA2(X_scaled)
        total = np.sum(S**2)
        explained = [np.square(si) / total for si in S]
        print(explained)

        W_12 = Vt.T[:, :2]
        proj_2D = X_mc.dot(W_12)

        pca = pd.DataFrame(proj_2D[:,0], columns=["PC1"])
        pca["PC2"] = proj_2D[:,1]
        pca["Y"] = labels
        print(pca.head())

        # plt.figure(figsize=(20, 10))
        #sns.scatterplot(x=res["PC1"], y=np.zeros(y.shape[0]), hue=res["Y"], s=200)
        sns.scatterplot(x=pca["PC1"], y=pca["PC2"], hue=pca["Y"], s=200)
    plt.show()
