import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from .PCA import PCA

def show():
    df = pd.read_csv("https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv")
    print(df.head())
    nda = df.to_numpy()
    X = nda[:,:-1]
    y = nda[:, -1]
    print(y.shape)

    eigvals, eigvecs, expl_variances = PCA(X)
    print(eigvals)
    print(eigvecs)
    print(expl_variances)

    X_mc = StandardScaler().fit_transform(X)
    proj_pc1 = X_mc.dot(eigvecs.T[0])
    proj_pc2 = X_mc.dot(eigvecs.T[1])

    res = pd.DataFrame(proj_pc1, columns=["PC1"])
    res["PC2"] = proj_pc2
    res["Y"] = y
    print(res.head())

    plt.figure(figsize=(20, 10))
    #sns.scatterplot(x=res["PC1"], y=np.zeros(y.shape[0]), hue=res["Y"], s=200)
    sns.scatterplot(x=res["PC1"], y=res["PC2"], hue=res["Y"], s=200)
    plt.show()
