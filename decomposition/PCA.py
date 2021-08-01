from sklearn.preprocessing import StandardScaler
import numpy as np

def PCA(X):
    X_mc = StandardScaler().fit_transform(X)
    X_features = X_mc.T
    covariance_matrix = np.cov(X_features)
    eigvals, eigvecs = np.linalg.eig(covariance_matrix)
    explained_variances = []

    by_total = 1 / np.sum(eigvals)
    for i in range(len(eigvals)):
        explained_variances.append(eigvals[i] * by_total)

    return eigvals, eigvecs, explained_variances
