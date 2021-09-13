import numpy as np
from cvxopt import matrix, solvers

class SoftMarginSVM:
    def __init__(self, N, C, X0, X1, X, y):
        self.N = N
        self. C = C
        self.X0 = X0
        self.X1 = X1
        self.X = X
        self.y = y

        self.X0_bar = np.vstack((X0.T, np.ones((1, N)))) # extended data
        self.X1_bar = np.vstack((X1.T, np.ones((1, N)))) # extended data
        self.Z = np.hstack((X0_bar, -X1_bar))
        self.lam = 1./C

    def duality_solve(self):
        V = np.concatenate((self.X0.T, -self.X1.T), axis=1)
        K = matrix(V.T.dot(V))

        p = matrix(-np.ones((2*self.N, 1)))
        G = matrix(np.vstack((-np.eye(2*self.N), np.eye(2*self.N))))
        h = matrix(np.vstack((np.zeros((2*self.N, 1)), self.C*np.ones((2*self.N, 1)))))
        A = matrix(self.y.reshape((-1, 2*self.N)))
        b = matrix(np.zeros((1, 1)))
        solvers.options['show_progress'] = False
        sol = solvers.qp(K, p, G, h, A, b)
        l = np.array(sol['x'])

        S = np.where(l > 1e-5)[0]
        S2 = np.where(l < .999*self.C)[0]
        M = [val for val in S if val in S2]

        XT = self.X.T
        VS = V[:, S]
        lS = l[S]
        yM = y[M]
        XM = XT[:, M]

        w_dual = VS.dot(lS).reshape(-1, 1)
        b_dual = np.mean(yM.T - w_dual.T.dot(XM))
        return (w_dual.T, b_dual)

    def cost(self, w):
        u = w.T.dot(self.Z)
        return np.sum(np.maximum(0, 1-u)) + .5*self.lam*np.sum(w*w) - .5*self.lam*w[-1]*w[-1]

    def gradient(self, w):
        u = w.T.dot(self.Z)
        H = np.where(u < 1)[1]
        ZS = self.Z[:, H]
        g = (-np.sum(ZS, axis=1, keepdims=True) + self.lam*w)
        g[-1] -= self.lam*w[-1]
        return g

    def gd(self, w0, eta):
        w = w0
        i = 0
        while i<100000:
            i = i+1
            g = self.gradient(w)
            w -= eta*g
            if (i%10000 == 1):
                print('iter %d' %i + ' cost: %f' %self.cost(w))
            if (np.linalg.norm(g) < 1e-5):
                break
        return w

    def unconstraint_opt_solve(self):
        w0 = np.random.randn(self.X0_bar.shape[0], 1)
        w = self.gd(w0, eta=0.001)
        w_hinge = w[:-1].reshape(-1, 1)
        b_hinge = w[-1]
        return w_hinge.T, b_hinge
