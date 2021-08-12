from sklearn.preprocessing import StandardScaler
import numpy as np

def PCA(X):
    # X_features:
    # each row = a feature
    # each col = an obs
    X_features = X.T

    # cov matrix: pxp
    covariance_matrix = np.cov(X_features)

    # each eigvec: px1
    # all eigvecs: pxp
    eigvals, eigvecs = np.linalg.eig(covariance_matrix)

    # diag matrix of eigvals pxp
    explained_variances = []

    by_total = 1 / np.sum(eigvals)
    for i in range(len(eigvals)):
        explained_variances.append(eigvals[i] * by_total)

    return eigvals, eigvecs, explained_variances

def calc_major_component(x, q, eigvecs, eigvals, explained_variances):
    # obs x
    # major threshold q < p
    # PCA of X
    pass

def calc_minor_component(x, r, eigvecs, eigvals, explained_variances):
    # obs x
    # major threshold q < p
    # PCA of X
    pass

def preprocess_anomaly_metrics(X):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    eigvals, eigvecs, explained = PCA(X)

    # PREPROCESS
    # calculate p PCs for each obs Xj of X_feat:
    # pci = eigveci.T dot Xj -> Xj has pc vector px1
    # -> PC matrix pxn of X

    # SELECT C1 THRESHOLD:
    # find q = # of first principle components explaining 50% variance, using eigvals
    # calculate major metric for each Xj: sigma(yi**2 / lambdai) i: 1 -> q
    # sort by metrics, desc, take 95% percentile

    # SELECT C2 THRESHOLD:
    # find r: eigvals lambda i: r+1 -> p: lambdai < 0.2
    # calculate minor metric for each Xj: sigma(yi**2 / lambdai) i: r+1 -> p
    # sort by metrics, desc, take 95% percentile

    # return c1, c2

    pass

def anomaly_classification(x0, c1, c2, eigvecs, eigvals):
    # CLASSIFICATION for x0
    # get p pcs of x0: pci = eigveci.T dot x0 i: 1 -> p

    # calculate major component metric sigma(yi**2 / lambdai) i: 1 -> q
    # calculate minor components metric: sigma(yi**2 / lambdai) i: r+1 -> p
    # decision anomaly if major > c1 or minor > c2
    # otherwise normal

    # return classification label: 1 if normal, -1 if anomaly

    pass

def PCA2(X):
    X_mc = X - X.mean(axis=0)
    U,S,Vt = np.linalg.svd(X_mc)
    total = np.sum(S**2)
    explained = [np.square(si) / total for si in S]

    return X_mc,explained,U,S,Vt
