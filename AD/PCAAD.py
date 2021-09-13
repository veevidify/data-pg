from sklearn.preprocessing import StandardScaler
import numpy as np

# todo: refactor this into the class
def PCA(X):
    X_mc = X - X.mean(axis=0)
    U,S,Vt = np.linalg.svd(X_mc)

    S_sq = S**2
    idx = np.argsort(S_sq)[::-1]
    eigenvalues = S_sq[idx]
    eigenvectors = Vt.T[:,idx]

    by_total = 1/np.sum(S_sq)
    explained = [si_sq * by_total for si_sq in S_sq]

    return eigenvalues, eigenvectors, explained

class PCAAD:
    def __init__(self, contamination=0.05):
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
        self.decision_scores_ = None

        self.contamination = contamination

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
        print(X.shape)
        # scaler = StandardScaler()
        # self.X = scaler.fit_transform(X)
        self.X = X
        print('==> X')
        print(self.X.shape)

        self.eigvals, self.eigvecs, explained = PCA(self.X)
        print('==> PCA')
        print(self.eigvals)
        print(explained)

        # PREPROCESS
        # calculate p PCs for each obs Xj of X_feat:
        # -> PC matrix pxn of X
        self.X_mean = self.X.mean(axis=0).reshape((1, -1)).T # mean of each feat, as col vec px1
        print('==> feature means')
        print(self.X_mean)

        self.X_feat = self.X.T # now cols = obs, rows = feats -> pxn
        print('==> features')
        print(self.X_feat.shape)
        X_PC = self.calc_principle_components(self.X_feat) # pxn
        print('==> PC-projected values')
        print(X_PC.shape)
        # print(X_PC)

        # c1 & c2 threshold satisfy:
        # if major metric <= c1 && minor metric <= c2 -> normal
        # if major matric > c1 || minor metric > c2 -> anomaly

        # SELECT C1 THRESHOLD:
        # find q = # of first principle components explaining 50% variance, using explained_variances
        # calculate major metric for each Xj
        # sort by metrics, desc, take 95% percentile
        sum_variances = 0
        n = self.X_feat.shape[1]
        p = self.X_feat.shape[0]
        for i in range(p):
            sum_variances = sum_variances + explained[i]
            if sum_variances > 0.5:
                self.q = i+1
                break
        major_metrics = self.calc_major_metrics(X_PC)
        print('==> major')
        print(self.q)

        outlier_percentage = self.contamination / 2
        # quantile
        self.c1 = np.quantile(major_metrics, 1-outlier_percentage) # 10% outliers from major metrics
        print(self.c1)
        # print(major_metrics)

        # SELECT C2 THRESHOLD:
        # find r: eigvals lambda i: r+1 -> p: lambdai < 0.2
        # calculate minor metric for each Xj
        # sort by metrics, desc, take 95% percentile
        self.r = np.searchsorted(self.eigvals, 0.2)
        minor_metrics = self.calc_minor_metrics(X_PC)
        print('==> minor')
        print(self.r)
        self.c2 = np.quantile(minor_metrics, 1-outlier_percentage) # 10% outliers from minor metrics
        print(self.c2)
        # print(minor_metrics)

        major_decision = (major_metrics - self.c1)
        minor_decision = (minor_metrics - self.c2)
        self.decision_scores_ = 1.0*(major_decision + abs(major_decision) + minor_decision + abs(minor_decision))

    def predict(self, X0):
        # CLASSIFICATION for X0 - matrix of new obs
        # get feats X0.T - col is obs, row is feat
        # get p PCs of each X0j: pci = eigveci.T dot X0j i: 1 -> p
        # calculate major components metric: sigma(yi**2 / lambdai) i: 1 -> q
        # calculate minor components metric: sigma(yi**2 / lambdai) i: r+1 -> p
        X0_feat = X0.T
        X0_PC = self.calc_principle_components(X0_feat)
        X0_major = self.calc_major_metrics(X0_PC)
        X0_minor = self.calc_minor_metrics(X0_PC)

        # decision: anomaly if major > c1 or minor > c2
        # normal otherwise
        n = X0_feat.shape[1]
        labels = np.zeros(n, dtype=int)
        for j in range(n):
            # if (X0_minor[j] > self.c2):
            # if (X0_major[j] > self.c1):
            if (X0_major[j] > self.c1 or X0_minor[j] > self.c2):
                labels[j] = 1

        # return classification labels for each x: 1 if normal, -1 if anomaly
        return labels

    def decision_function(self, X0):
        X0_feat = X0.T
        X0_PC = self.calc_principle_components(X0_feat)
        X0_major = self.calc_major_metrics(X0_PC)
        X0_minor = self.calc_minor_metrics(X0_PC)

        major_decision = (X0_major - self.c1)
        minor_decision = (X0_minor - self.c2)
        return 1.0*(major_decision + abs(major_decision) + minor_decision + abs(minor_decision))
        # return X0_major**2 / self.c1**2 + X0_minor**2 / self.c2**2

