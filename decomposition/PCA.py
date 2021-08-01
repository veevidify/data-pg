from sklearn.preprocessing import StandardScaler
import numpy as np

def PCA(X):
    X_features = X_scaled.T
    covariance_matrix = np.cov(X_features)
    eigvals, eigvecs = np.linalg.eig(covariance_matrix)
    explained_variances = []

    by_total = 1 / np.sum(eigvals)
    for i in range(len(eigvals)):
        explained_variances.append(eigvals[i] * by_total)

    return eigvals, eigvecs, explained_variances

def PCA2(X):
    X_mc = X - X.mean(axis=0)
    U,S,V_T = np.linalg.svd(X_mc)

    return X_mc,U,S,V_T
