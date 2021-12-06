import numpy as np

# based on tutorial https://nlml.github.io/in-raw-numpy/in-raw-numpy-t-sne/
# treating xi as column
# has to transpose from input since input's row = obs

# Dij = -|xi - xj|^2
def n_sq_euclidean(X):
    # xi^Txi - 2xi^Txj + xj^Txj
    quad_X = np.sum(np.square(X))
    D = np.add(
        np.add(
            -2*np.dot(X.T, X),
            quad_X
        ), quad_X
    )
    return -D

def weighted_by_total(X, zero_diag=True):
    pass

# regular SNE formulas:
# p j|i = exp(-|xi - xj|^2 / (2s^2)) / sum_k!=i
# q j|i = exp(-|yi-yj|^2) / sum_k!=i
def pdf_matrix(dist, sigmas=None):
    pass

# perp(pi) = 2^(H(pi))
# H(pi) = - sum_j pj|i log2(pj|i)
def perp(pdf_matrix):
    pass

# binary search to find desired perplexity
def bsearch_sigma(eval_f, target, precision=1e-10, max_iter=1e5, lb=1e-20, ub=1e3):
    pass

def find_optimal_sigma(dists, target_perp):
    pass

def get_perp(dists, sigmas):
    pass

def joint_q(Y):
    pass

def joint_p(Y):
    pass

def joint_p_from_cond_p(P):
    pass

def tsne(X):
    pass

