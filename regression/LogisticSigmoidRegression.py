import numpy as np

class LogisticSigmoidRegression:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def sigmoid(self, s):
        return 1 / (1 + np.exp(-s))

    def fit(self, w_init, eta, tol=1e-4, max_count=100000):
        w = [w_init]
        i = 0
        N = self.X.shape[1]
        d = self.X.shape[0]
        count = 0
        check_w_after = 20
        while count < max_count:
            mix_id = np.random.permutation(N)
            for i in mix_id:
                xi = self.X[:, i].reshape(d, 1)
                yi = self.y[i]
                zi = self.sigmoid(np.dot(w[-1].T, xi))
                w_new = w[-1] + eta*(yi - zi)*xi
                count += 1
                if (count % check_w_after == 0):
                    if (np.linalg.norm(w_new - w[-check_w_after]) < tol):
                        return w
                w.append(w_new)
        return w

