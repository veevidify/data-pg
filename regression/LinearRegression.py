import numpy as np

class LinearRegression:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
        self.X_bar = np.concatenate(
            (np.ones((self.X.shape[0], 1)), self.X),
            axis = 1)
        
        self.N = self.X_bar.shape[0]
    
    # least square linear regression
    def fit(self):
        # Least square projection formula
        A = np.dot(self.X_bar.T, self.X_bar)
        b = np.dot(self.X_bar.T, self.y)
        w = np.dot(np.linalg.pinv(A), b) # X_bar^TX_barw = X_bar^Ty

        return w

    # gd
    def cost(self, w):
        # 1/2 * N * ||y-Xw||_2^2
        return 0.5*self.N*np.linalg.norm(self.y - self.X_bar.dot(w), 2)**2

    def cost_gradient(self, w):
        # X_bar^TX_barw = X_bar^Ty
        return 1/self.N * self.X_bar.T.dot(self.X_bar.dot(w) - self.y)

    def gd_converged(self, w):
        return np.linalg.norm(self.cost_gradient(w)) / len(w) < 1e-3

    def gd(self, w_init, eta):
        w = [w_init]
        for i in range(500):
            w_descent = w[-1] - eta*self.cost_gradient(w[-1])
            if (self.gd_converged(w_descent)):
                break
            w.append(w_descent)
        
        return (w, i)

    # gd with momentum
    def gdm(self, theta_init, eta, gamma):
        theta = [theta_init]
        v = np.zeros_like(theta_init)
        for i in range(500):
            v_new = gamma*v + eta*self.cost_gradient(theta[-1])
            theta_descent = theta[-1] - v_new
            if (self.gd_converged(theta_descent)):
                break
            theta.append(theta_descent)
            v = v_new

        return (theta, i)

    # stochastic gd
    def sgd_converged(self, w, w_prev, w_init):
        return np.linalg.norm(w - w_prev) / len(w_init) < 1e-3

    def sgrad(self, w, i, rd_id):
        true_i = rd_id[i]
        xi = self.X_bar[true_i, :]
        yi = self.y[true_i]

        a = np.dot(w.T, xi) - yi

        return (xi*a).reshape(2, 1)

    def sgd(self, w_init, eta):
        w = [w_init]
        w_last_check = w_init
        i_check_w = 10
        N = self.X.shape[0]
        count = 0
        for ite in range(10):
            rd_id = np.random.permutation(N)
            for i in range(N):
                count += 1
                g = self.sgrad(w[-1], i, rd_id)
                w_new = w[-1] - eta*g
                w.append(w_new)
                if (count % i_check_w == 0):
                    w_this_check = w_new
                    if (self.sgd_converged(w_this_check, w_last_check, w_init)):
                        return w
                    w_last_check = w_this_check
        return w
