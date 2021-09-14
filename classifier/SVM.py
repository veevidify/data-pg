import numpy as np
from cvxopt import matrix, solvers

class SVM:
    def __init__(self, X0, X1):
        N = X0.shape[0]
        self.N = N
        self.X0 = X0
        self.X1 = X1
        self.X = np.concatenate((X0.T, X1.T), axis=1)
        self.y = np.concatenate((np.ones((1, N)), -1*np.ones((1, N))), axis=1)

        self.S = None
        self.w = None
        self.b = None

    def fit(self):
        # build K
        V = np.concatenate((self.X0.T, -self.X1.T), axis = 1)
        K = matrix(V.T.dot(V))

        p = matrix(-np.ones((2*self.N, 1))) # all-one vector

        # build A, b, G, h
        G = matrix(-np.eye(2*self.N)) # for all lambda_n >= 0
        h = matrix(np.zeros((2*self.N, 1)))
        A = matrix(self.y) # the equality constrain is actually y^T lambda = 0
        b = matrix(np.zeros((1, 1)))

        solvers.options['show_progress'] = False
        sol = solvers.qp(K, p, G, h, A, b)

        l = np.array(sol['x'])
        print('lambda = ')
        print(l.T)
        
        # support vectors
        epsilon = 1e-6 # just a small number, greater than 1e-9
        self.S = np.where(l > epsilon)[0] # support vector indices

        # support vectors
        XS = self.X[:, self.S]
        yS = self.y[:, self.S]
        VS = V[:, self.S]
        lS = l[self.S]

        # calculate w and b
        self.w = VS.dot(lS)
        self.b = np.mean(yS.T - self.w.T.dot(XS))
        return self
    
    def get_hyperplane(self):
        return self.w, self.b
        