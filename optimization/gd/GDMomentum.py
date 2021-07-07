import numpy as np

def has_converged(theta, gradient):
    return np.linalg.norm(gradient(theta)) / len(theta) < 1e-3

def GDM(theta_init, gradient, eta, gamma):
    theta = [theta_init]
    v = np.zeros_like(theta_init)
    for i in range(100):
        v_new = gamma*v + eta*gradient(theta[-1])
        theta_new = theta[-1] - v_new
        if (has_converged(theta_new, gradient)):
            break
        theta.append(theta_new)
        v = v_new
    return theta
