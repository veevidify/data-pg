import numpy as np

from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import rbf_kernel

from cvxopt import matrix, solvers

from sklearn.model_selection import GridSearchCV, LeaveOneOut, KFold

# implementation based on https://github.com/lminvielle/mom-kde/blob/master/libs/kde_lib.py
class RKDE:
    def __init__(self, sigma=None, rho_type='hampel'):
        self.rho_type = rho_type

        self.sigma = sigma

        self.w = None
        self.X = None

    # kernel to calculate incoming datapoint's density estimate
    def gaussian(self, X):
        n, d = X.shape

        # 1/(2*pi*s^2)^(d/2) * exp(- X/(2s^2))
        K = np.exp(-X / (2*(self.sigma**2))) / ((2*np.pi*(self.sigma**2))**(d/2))
        return K

    def choose_sigma_via_cv(self, X, loo=False, kfold_splits=5):
        # search for appopriate sigma using CV on regular KDE
        b = np.logspace(-1.5, 0.5, 80)
        if (loo):
            grid_search = GridSearchCV(
                KernelDensity(kernel='gaussian'),
                {'bandwidth': b},
                cv=LeaveOneOut())
        else:
            grid_search = GridSearchCV(
                KernelDensity(kernel='gaussian'),
                {'bandwidth': b},
                cv=KFold(n_splits=kfold_splits))

        grid_search.fit(X)
        sigma = grid_search.best_params_['bandwidth']
        losses = grid_search.cv_results_['mean_test_score']
        return sigma, b, losses

    # (Huber, 1964)
    def rho(self, rho_type, x, a=0, b=0, c=0):
        if (rho_type == 'huber'):
            case1 = (x<=a)
            case2 = (x>a)
            t1 = x[case1]**2 / 2
            t2 = x[case2]*a - a**2 / 2
            L = np.sum(t1) + np.sum(t2)

        if (rho_type == 'hampel'):
            case1 = (x<a)
            case2 = np.logical_and(a<=x, x<b)
            case3 = np.logical_and(b<=x, x<c)
            case4 = (c<=x)

            t1 = x[case1]**2 / 2
            t2 = a*x[case2] - a**2 / 2
            t3 = (a * (x[case3]-c)**2 / (2*(b-c))) + a*(b+c-a)/2
            t4 = np.ones(x[case4].shape) * a * (b+c-a)/2

            L = np.sum(t1) + np.sum(t2) + np.sum(t3) + np.sum(t4)

        if (rho_type == 'sq'):
            L = np.sum(x**2)

        if (rho_type == 'abs'):
            L = np.sum(np.abs(x))

        return L/ x.shape[0]

    def loss(self, x, rho_type, a=0, b=0, c=0):
        return self.rho(rho_type=rho_type, x=x, a=a, b=b, c=c) / x.shape[0]

    # (Kim et. al, 2011)
    def psi(self, x, rho_type='hampel', a=0, b=0, c=0):
        if (rho_type == 'huber'):
            return np.minimum(x, a)

        if (rho_type == 'hampel'):
            case1 = (x<a)
            case2 = np.logical_and(a<=x, x<b)
            case3 = np.logical_and(b<=x, x<c)
            case4 = (c<=x)

            t1 = x[case1]
            t2 = np.ones(x[case2].shape) * a
            t3 = a * (c-x[case3]) / (c-b)
            t4 = np.zeros(x[case4].shape)

            ret = np.concatenate((t1, t2, t3, t4)).reshape((-1, x.shape[1]))
            return ret
        if (rho_type == 'sq'):
            return 2*x

        if (rho_type == 'abs'):
            return 1

    # (Kim et. al, 2011)
    def phi(self, x, rho_type='hampel', a=0, b=0, c=0):
        # adjust for divide by 0
        x[x==0.0] = 1e-5
        p = self.psi(x, rho_type=rho_type, a=a, b=b, c=c)
        return p / x

    # (Kim et. al, 2011)
    def kirwls(self, Ksigma, rho_type, n, a, b, c, alpha=1e-7, max_iter=100):
        # init weights
        w = np.ones((n, 1)) / n

        # first iter
        # ||Q(xj) - f(k)||
        # = <Q(xj), Q(xj)>H - 2<Q(xj), f(k)>H + <f(k), f(k)>H
        # = ks(xj, xj) - 2 Sigma_i wi.ks(xj, xi) + Sigma_i Sigma_l wi.wl.ks(xi, xl)
        st_term = np.diag(Ksigma).reshape((-1, 1))
        nd_term = -2 * np.dot(Ksigma, w)
        rd_term = np.dot(np.dot(w.T, Ksigma), w)
        s = st_term + nd_term + rd_term
        norm1 = np.sqrt(s)
        J = self.loss(x=norm1, rho_type=rho_type, a=a, b=b, c=c)

        # irls
        losses_seq = [J]
        loop = True
        i = 0
        while loop:
            J_prev = J

            w = self.phi(x=norm1, rho_type=rho_type, a=a, b=b, c=c)
            # normalizing to get Sigma_i wi = 1
            w = w / np.sum(w)

            st_term = np.diag(Ksigma).reshape((-1, 1))
            nd_term = -2 * np.dot(Ksigma, w)
            rd_term = np.dot(np.dot(w.T, Ksigma), w)
            s = st_term + nd_term + rd_term
            norm1 = np.sqrt(s)
            J = self.loss(x=norm1, rho_type=rho_type, a=a, b=b, c=c)
            losses_seq.append(J)

            # irls convergence
            if ((np.abs(J - J_prev) < (J_prev * alpha)) or (i == max_iter)):
                loop = False

            i = i+1

        return w, norm1, losses_seq

    def fit(self, X, alpha=1e-7, max_iter=100):
        self.X = X
        self.n, self.d = X.shape

        if (self.sigma == None):
            self.sigma, _, _ = self.choose_sigma_via_cv(self.X)

        gamma = 1.0 / (2 * (self.sigma**2))
        # gaussian kernel matrix
        print('== calculating Ks')
        Ksigma = rbf_kernel(X, X, gamma=gamma) * (2 * np.pi * self.sigma**2)**(-self.d/2.0)

        # run kirwls using norm1 abs loss to get set of good-enough a, b, c
        # (Huber, 1964)
        a = b = c = 0
        print('== finding a b c via kirwls with abs loss')
        w, norm, losses = self.kirwls(Ksigma, rho_type='abs', n=self.n, a=a, b=b, c=c, alpha=alpha, max_iter=max_iter)
        a = np.median(norm)
        b = np.percentile(norm, 75)
        c = np.percentile(norm, 95)

        # kirwls to model rkde
        print('== finding w via kirwls with hampel loss')
        w, norm, losses = self.kirwls(Ksigma, self.rho_type, self.n, a, b, c, alpha, max_iter)
        z = np.dot(Ksigma, w)

        self.w = w

        # return a collection of weighted kdes
        return z, w


    def score_samples(self, X):
        # f(x0) = sum_i wiQi (x0)

        n_pred = X.shape[0]
        Ksigma = np.zeros((n_pred, self.n))

        print('== obtaining Ks on x0')
        for d_i in range(self.d):
            # observations are rows, so receiving & reshaping to column
            temp_xpred = X[:, d_i].reshape((-1, 1))
            temp_x = self.X[:, d_i].reshape((-1, 1))
            # exp(xi - x0)
            Ksigma = Ksigma + (np.dot(np.ones((n_pred, 1)), temp_x.T) - np.dot(temp_xpred, np.ones((1, self.n))))**2

        Ksigma = self.gaussian(Ksigma)
        z = np.dot(Ksigma, self.w)
        return z[:, 0]
