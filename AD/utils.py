import numpy as np
import pandas as pd

from decomposition.PCA import PCA, PCA2

def get_pca(X):
    # X_scaled = StandardScaler().fit_transform(X)
    X_scaled = X

    X_mc,explained,U,S,Vt = PCA2(X_scaled)
    total = np.sum(S**2)
    explained = [np.square(si) / total for si in S]
    print(explained)

    W = Vt.T[:, :3]
    proj = X_mc.dot(W)

    pca = pd.DataFrame(proj[:,0], columns=["PC1"])
    pca["PC2"] = proj[:,1]
    pca["PC3"] = proj[:,2]

    return pca
