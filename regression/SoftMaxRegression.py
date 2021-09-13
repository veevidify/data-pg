import numpy as np
from scipy import sparse

class SoftMaxRegression:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def softmax_stable(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims = True))
        A = exp_Z / exp_Z.sum(axis=0)
        return A

    def convert_labels(self, y, C):
        Y = sparse.coo_matrix(
            (np.ones_like(y), (y, np.arange(len(y)))),
            shape = (C, len(y))
        ).toarray()
        return Y

    def cost(self, X, Y, W):
        A = self.softmax_stable(W.T.dot(X))
        return -np.sum(Y*np.log(A))

    def grad(self, X, Y, W):
        A = self.softmax_stable(W.T.dot(X))
        E = A - Y
        return X.dot(E.T)

    def fit(self, W_init, eta, tol=1e-4, max_count=10000):
        W = [W_init]
        C = W_init.shape[1]
        Y = self.convert_labels(self.y, C)
        i = 0
        N = self.X.shape[1]
        d = self.X.shape[0]

        c = 0
        check_w_after = 20
        while c < max_count:
            mix_ids = np.random.permutation(N)
            for i in mix_ids:
                xi = self.X[:, i].reshape(d, 1)
                yi = Y[:, i].reshape(C, 1)
                ai = self.softmax_stable(np.dot(W[-1].T, xi))
                W_new = W[-1] + eta*xi.dot((yi - ai).T)
                c += 1

                if (c % check_w_after == 0):
                    if (np.linalg.norm(W_new - W[-check_w_after]) < tol):
                        return W

                W.append(W_new)
        return W
