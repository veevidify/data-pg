from sklearn.preprocessing import StandardScaler
import numpy as np

from decomposition.PCA import PCA

class PCAAD:
    def __init__():
        # nxp: each row is an obs, each col is a feat, standard-scaled
        self.X = None

        # X.T -> pxn
        self.X_feat = None

        # px1 is mean vector of orig matrix nxp
        self.X_mean = None

        # eigvecs is pxp of orig matrix, each col is an eig vec
        self.eigvecs = None

        # eigvals of orig X
        self.eigvals = None

        # major component calculation q & decision c1
        self.q = None
        self.c1 = None

        # minor component calculation r & decision c2
        self.r = None
        self.c2 = None

    def calc_principle_components(self, X):
        # X is col vec of an obs or matrix of col vecs pxn, each col is an obs (from input X or new obs X0)

        X_mc = X - self.X_mean
        # pci = eigveci.T dot Xj -> Xj has pc vector px1
        return self.eigvecs.T.dot(X_mc)

    def calc_major_metrics(self, X_PC):
        # obs matrix's PCA (cols as obs) pxn (from input X or new obs X0)
        # major threshold q < p

        # each col Xj: sigma(pci**2 / lambdai) i: 1 -> q
        # equiv. mahalanobis distance in the first q
        n = X_PC.shape[1]
        metrics = np.zeros(n) # init empty array of metric for each obs
        for j in range(n):
            s = 0
            for i in range(self.q):
                s = s + X_PC[i][j]**2 / self.eigvals[i]
            metrics[j] = s

        return metrics

    def calc_minor_metrics(self, X_PC):
        # obs matrix's PCA (cols as obs) pxn (from input X or new obs X0)
        # minor threshold r < p

        # each col Xj: sigma(pci**2 / lambdai) i: r+1 -> p
        # equiv. mahalanobis distance in the last p-r
        n = X_PC.shape[1]
        p = X_PC.shape[0]
        metrics = np.zeros(n) # init empty array of metric for each obs
        for j in range(n):
            s = 0
            for i in range(self.r+1, p):
                s = s + X_PC[i][j]**2 / self.eigvals[i]
            metrics[j] = s

        return metrics

    def fit(self, X):
        scaler = StandardScaler()
        self.X = scaler.fit_transform(X)

        self.eigvals, self.eigvecs, self.explained = PCA(self.X)

        # PREPROCESS
        # calculate p PCs for each obs Xj of X_feat:
        # -> PC matrix pxn of X
        self.X_mean = self.X.mean(axis=0).reshape((1, -1)) # mean of each feat, as col vec px1
        self.X_feat = self.X.T # now cols = obs, rows = feats -> pxn
        X_PC = self.calc_principle_components(self.X_feat) # pxn

        # SELECT C1 THRESHOLD:
        # find q = # of first principle components explaining 50% variance, using explained_variances
        # calculate major metric for each Xj
        # sort by metrics, desc, take 95% percentile
        sum_variances = 0
        n = self.X_feat.shape[1]
        p = self.X_feat.shape[0]
        for i in range(p):
            sum_variances = sum_variances + explained[i]
            if sum > 0.5:
                self.q = i
                break
        major_metrics = self.calc_major_metrics(X_PC)
        # quantile
        self.c1 = np.quantile(major_metrics, 0.05) # 5% outliers

        # SELECT C2 THRESHOLD:
        # find r: eigvals lambda i: r+1 -> p: lambdai < 0.2
        # calculate minor metric for each Xj
        # sort by metrics, desc, take 95% percentile
        self.r = np.searchsorted(eigvals, 0.2)
        minor_metrics = self.calc_minor_metrics(X_PC)
        self.c2 = np.quantile(minor_metrics, 0.05) # 5% outliers

    def predict(self, X0):
        # CLASSIFICATION for X0 - matrix of new obs
        # get feats X0.T - col is obs, row is feat
        # get p PCs of each X0j: pci = eigveci.T dot X0j i: 1 -> p
        # calculate major components metric: sigma(yi**2 / lambdai) i: 1 -> q
        # calculate minor components metric: sigma(yi**2 / lambdai) i: r+1 -> p
        X0_feat = X0.T
        X0_PC = self.calc_principle_components(X0_feat)
        X0_major = self.calc_major_metrics(X0_pc)
        X0_minor = self.calc_major_metrics(X0_pc)

        # decision: anomaly if major > c1 or minor > c2
        # normal otherwise
        n = X0.feat.shape[1]
        labels = np.ones(n)
        for j in range(n):
            if (X0_major[j] > c1 or X0_minor[j] > c2):
                labels[j] = -1

        # return classification labels for each x: 1 if normal, -1 if anomaly
        return labels
