from sklearn.preprocessing import StandardScaler
import numpy as np

def PCA(X):
    X_features = X.T
    covariance_matrix = np.cov(X_features)
    eigvals, eigvecs = np.linalg.eig(covariance_matrix)
    explained_variances = []

    by_total = 1 / np.sum(eigvals)
    for i in range(len(eigvals)):
        explained_variances.append(eigvals[i] * by_total)

    return eigvals, eigvecs, explained_variances

def PCA2(X):
    X_mc = X - X.mean(axis=0)
    U,S,Vt = np.linalg.svd(X_mc)
    total = np.sum(S**2)
    explained = [np.square(si) / total for si in S]

    return X_mc,explained,U,S,Vt
