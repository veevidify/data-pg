import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from .PCA import PCA, PCA2

def show():
    df = pd.read_csv("https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv")
    print(df.head())
    nda = df.to_numpy()
    X = nda[:,:-1]
    y = nda[:, -1]
    print(y.shape)

    X_scaled = StandardScaler().fit_transform(X)
    eigvals, eigvecs, expl_variances = PCA(X_scaled)
    print(eigvals)
    print(eigvecs)
    print(expl_variances)

    proj_pc1 = X_scaled.dot(eigvecs.T[0])
    proj_pc2 = X_scaled.dot(eigvecs.T[1])

    res = pd.DataFrame(proj_pc1, columns=["PC1"])
    res["PC2"] = proj_pc2
    res["Y"] = y
    print(res.head())

    plt.figure(figsize=(20, 10))
    #sns.scatterplot(x=res["PC1"], y=np.zeros(y.shape[0]), hue=res["Y"], s=200)
    sns.scatterplot(x=res["PC1"], y=res["PC2"], hue=res["Y"], s=200)
    plt.show()

def show2():
    df = pd.read_csv("https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv")
    print(df.head())
    nda = df.to_numpy()
    X = nda[:,:-1]
    y = nda[:, -1]
    print(y.shape)

    X_scaled = StandardScaler().fit_transform(X)
    X_mc,explained,U,S,Vt = PCA2(X_scaled)
    total = np.sum(S**2)
    explained = [np.square(si) / total for si in S]
    print(explained)

    W_12 = Vt.T[:, :2]
    proj_2D = X_mc.dot(W_12)

    res = pd.DataFrame(proj_2D[:,0], columns=["PC1"])
    res["PC2"] = proj_2D[:,1]
    res["Y"] = y
    print(res.head())

    plt.figure(figsize=(20, 10))
    #sns.scatterplot(x=res["PC1"], y=np.zeros(y.shape[0]), hue=res["Y"], s=200)
    sns.scatterplot(x=res["PC1"], y=res["PC2"], hue=res["Y"], s=200)
    plt.show()
