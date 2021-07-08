import numpy as np

class StochasticGD:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def sgrad(self, w, i, rd_id):
        true_i = rd_id[i]
        xi = self.X[true_i, :]
        yi = self.y[true_i]
        a = np.dot(xi, w) - yi
        return (xi*a).reshape(2, 1)

    def stochastic_gradient_descent(self, w_init, eta):
        w = [w_init]
        w_last_check = w_init
        i_check_w = 10
        N = self.X.shape[0]
        count = 0
        for i in range(10):
            rd_id = np.random.permutation(N)
            for i in range(N):
                count += 1
                g = self.sgrad(w[-1], i, rd_id)
                w_new = w[-1] - eta*g
                w.append(w_new)
                if (count % i_check_w == 0):
                    w_this_check = w_new
                    if (np.linalg.norm(w_this_check - w_last_check) / len(w_init) < 1e-3):
                        return w
                    w_last_check = w_this_check
        return w

