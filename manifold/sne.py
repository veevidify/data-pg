import numpy as np

# based on tutorial https://nlml.github.io/in-raw-numpy/in-raw-numpy-t-sne/
# treating row xi as obs ~~ sklearn convention

class SNE:
    def __init__(self, algo='TSNE', perplexity=40, descent_iterations=400, descent_momentum=0.9, learning_rate=10.):
        if (algo != 'TSNE' and algo != 'SSNE'):
            raise ValueError("Invalid algo. Accept 'TSNE' & 'SSNE'")

        self.algo = algo
        self.desired_perp = perplexity
        self.gd_iters = descent_iterations
        self.momentum = descent_momentum
        self.alpha = learning_rate

        self.rng = np.random.RandomState(1)

    # Dij = -|xi - xj|^2
    def n_sq_euclidean(self, X):
        # xi as row-vec
        # xixi^T - 2xixj^T + xjxj^T
        quad_X = np.sum(np.square(X), axis=1)
        D = np.add(
            np.add(
                -2*np.dot(X, X.T),
                quad_X
            ).T,
            quad_X
        )
        return -D

    def weighted_by_total(self, X, zero_diag=True, index=None):
        # numerical stability each elem -=max_of_row
        exp_X = np.exp(X - np.max(X, axis=1).reshape([-1, 1]))

        if (zero_diag and index is None):
            np.fill_diagonal(exp_X, 0)

        # single entry or entire matrix
        if (index is not None):
            exp_X[:, index] = 0.

        # numerical stability += eps (for log function)
        exp_X = exp_X + 1e-8

        # return each elem / total
        return exp_X / exp_X.sum(axis=1).reshape([-1, 1])

    # regular SNE formulas:
    # p j|i = exp(-|xi - xj|^2 / (2s^2)) / sum_k!=i
    # q j|i = exp(-|yi-yj|^2) / sum_k!=i
    def pdf_matrix(self, euclidean_dists, sigmas=None, index=None):
        if (sigmas is not None):
            # 2s^2
            ss2 = 2. * np.square(sigmas.reshape([-1, 1]))
        else:
            ss2 = 1

        return self.weighted_by_total(euclidean_dists / ss2, index=index)

    # perp(pi) = 2^(H(pi))
    # H(pi) = - sum_j pj|i log2(pj|i)
    def perp(self, pdf_matrix):
        # row-wise shannon entropy score H(pi)
        H_pi = -np.sum(pdf_matrix * np.log2(pdf_matrix), axis=1)
        perp = 2 ** H_pi
        return perp

    def get_perp(self, euclidean_dists, sigmas, index):
        return self.perp(self.pdf_matrix(euclidean_dists, sigmas, index))

    # binary search to find desired perplexity
    def bsearch_sigma(self, f, target, precision=1e-7, max_iter=1000, lb=1e-20, ub=1e3):
        for i in range(max_iter):
            mid = (lb + ub) / 2
            mid_val = f(mid)

            # left half
            if (target < mid_val):
                ub = mid
            # right half
            else:
                lb = mid

            if (np.abs(mid_val - target) <= precision):
                break

        return mid

    def find_optimal_sigmas(self, euclidean_dists, target_perp):
        # find s_i for each row
        sigmas = []
        n = euclidean_dists.shape[0]
        for i in range(n):
            si = self.bsearch_sigma(
                lambda s: self.get_perp(euclidean_dists[i:i+1, :], np.array(s), index=i),
                target=target_perp
            )

            sigmas.append(si)

        return np.array(sigmas)

    # t-SNE: calc symmetric p_ij from p j|i
    # p_ij = p_ji = (p i|j + p j|i) / 2n
    # get final P from input matrix X
    def joint_p(self, X, target_perp):
        euclidean = self.n_sq_euclidean(X)
        sigmas = self.find_optimal_sigmas(euclidean, target_perp)
        cond_P = self.pdf_matrix(euclidean, sigmas=sigmas)

        n = cond_P.shape[0]
        sym_P = (cond_P + cond_P.T) / (2*n)

        return sym_P

    # symmetric SNE: calc symmetric q_ij instead of using q j|i
    # q_ij = q_ji = exp(-|yi-yj|^2) / sum_k!=l exp(-|yk-yl|^2)
    def joint_q_ssne(self, Y):
        n_s_euclidean = self.n_sq_euclidean(Y)
        exp_euclidean = np.exp(n_s_euclidean)
        np.fill_diagonal(exp_euclidean, 0.)

        return exp_euclidean / np.sum(exp_euclidean), None

    # dC / dyi = 4 sum_j (pij - qij)(yi - yj)
    def ssne_grad(self, P, Q, Y, euclidean=None):
        # pij - qij
        pq_diff = P - Q

        # numpy tricks
        # n vertical stacks, each stack is pi - qi (n rows / nxnx1)
        pq_stack = np.expand_dims(pq_diff, 2)

        # this create n versions
        # each version j is entire y - yj, nxnx2
        y_diffs = np.expand_dims(Y, 1) - np.expand_dims(Y, 0)

        # the first n is for the whole dataset,
        # which gradient uses for each obs i
        # we basically mult entry-wise pi-qi with y-yi,
        # then call sum to aggregate them (sum_j)
        grad_val = 4. * (pq_stack * y_diffs).sum(axis=1)

        return grad_val

    # t-dist SNE: calc student-t 1deg of freedom q to model y
    # q_ij = q_ji = (1+|yi-yj|^2)^-1 / sum_k!=l (1+|yk-yl|^2)^-1
    def joint_q_tsne(self, Y):
        n_s_euclidean = self.n_sq_euclidean(Y)
        euclidean_inv = np.power(1. - n_s_euclidean, -1)
        np.fill_diagonal(euclidean_inv, 0.)

        return euclidean_inv / np.sum(euclidean_inv), euclidean_inv

    # dC / dyi = 4 sum_j (pij-qij)(yi-yj)(1+|yi-yj|^2)^-1
    def tsne_grad(self, P, Q, Y, euclidean_inv):
        # similar tricks to symmetric grad calculation
        # pij - qij
        pq_diff = P - Q

        # numpy tricks
        # n vertical stacks, each stack is pi - qi (n rows / nxnx1)
        pq_stack = np.expand_dims(pq_diff, 2)

        # this create n versions
        # each version j is entire y - yj, nxnx2
        y_diffs = np.expand_dims(Y, 1) - np.expand_dims(Y, 0)

        euclidean_stacks = np.expand_dims(euclidean_inv, 2)

        # the first n is for the whole dataset,
        # which gradient uses for each obs i
        # we basically mult entry-wise pi-qi with y-yi,
        # then call sum to aggregate them (sum_j)
        y_diffs_euclidean = y_diffs * euclidean_stacks
        grad_val = 4. * (pq_stack * y_diffs_euclidean).sum(axis=1)

        return grad_val

    def gd_momentum(self, X, P, max_iter, q_fn, grad_fn, alpha, m):
        n = X.shape[0]
        # Init a random repr: Y to start optimizing
        Y = self.rng.normal(0., 1e-4, [n, 2])

        # momentum descent: Y(k+1) = Y(k) - a*grad + m(Y(k) - Y(k-1))
        Yk_1 = Y.copy()
        Yk = Y.copy()

        for k in range(max_iter):
            if ((k+1) % 100 == 0):
                print("== GDM - iter == ", k+1)
            Q, euclidean = q_fn(Y)

            grad_val = grad_fn(P, Q, Y, euclidean)
            Y = Y - alpha*grad_val + m*(Yk - Yk_1)

            Yk_1 = Yk.copy()
            Yk = Y.copy()

        return Y

    def ssne_fit(self, X):
        print("==> s-SNE")
        print("== init P")
        P = self.joint_p(X, target_perp=self.desired_perp)
        print("== GDM")
        Y = self.gd_momentum(
            X=X,
            P=P,
            max_iter=self.gd_iters,
            q_fn=self.joint_q_ssne,
            grad_fn=self.ssne_grad,
            alpha=self.alpha,
            m=self.momentum
        )
        print("<==")
        return Y

    def tsne_fit(self, X):
        print("==> t-SNE")
        print("== init P")
        P = self.joint_p(X, target_perp=self.desired_perp)
        print("== GDM")
        Y = self.gd_momentum(
            X=X,
            P=P,
            max_iter=self.gd_iters,
            q_fn=self.joint_q_tsne,
            grad_fn=self.tsne_grad,
            alpha=self.alpha,
            m=self.momentum
        )
        print("<==")
        return Y

    def fit(self, X):
        if (self.algo == 'SSNE'):
            return self.ssne_fit(X)
        elif (self.algo == 'TSNE'):
            return self.tsne_fit(X)
