import math
import numpy as np

class LogisticModelIRLS:
    def __init__(self, X, y, varnames):
        self.varnames = varnames
        self.X = X
        self.y = y

        self.converged = False
        self.converged_at = -1
        self.nll_seq = []
        self.beta = None

    def nll(self, beta, h, p):
        p_adj = p

        # adjust so no ln of 0
        p_adj[p_adj == 1.0] = 1 - 1e-6
        p_adj[p_adj == 0] = 1e-6

        # - Sigma yi.ln(p) + (1-yi).ln(1-p)
        neg_ln = - ((1 - self.y).dot(np.log(1 - p_adj)) + self.y.dot(np.log(p_adj)))
        return neg_ln

    def wls(self, beta, h, p):
        w = p * (1-p)
        W = np.diag(w)
        arbitrary_small = np.ones_like(w, dtype='float64') * 1e-6

        # working responses
        z = h + np.divide(self.y - p, w, out=arbitrary_small, where=w!=0)

        # least square Newton-Raphson update
        XT = np.transpose(self.X)
        XTWX = XT.dot(W).dot(self.X)
        XTWX_inv = np.linalg.inv(XTWX)
        XTWX_inv_XTW = XTWX_inv.dot(XT).dot(W)

        beta = XTWX_inv_XTW.dot(z)

        return beta

    def fit(self, iters=25):
        y_bar = np.mean(self.y)
        beta_init = math.log(y_bar / (1-y_bar))
        beta = np.array([0] * self.X.shape[1], dtype='float64')

        nll_seq = []
        self.converged = False
        for i in range(iters):
            betaTX = self.X.dot(beta)
            p = 1 / (1 + np.exp(-betaTX))

            nll = self.nll(beta, betaTX, p)
            nll_seq += [nll]

            if (i > 1):
                if (not self.converged and abs(nll_seq[-1] - nll_seq[-2]) < 1e-6):
                    self.converged = True
                    self.converged_at = i+1

            beta = self.wls(beta, betaTX, p)

        self.nll_seq = nll_seq
        self.beta = beta

        if not self.converged:
            print('IRLS failed to converge. Increase the number of iterations.')

        return self

    def info(self):
        if (not hasattr(self, 'beta')):
            print('Run fit first')
            return None

        coef_labels = ['---------------','<Intercept>']+list(self.varnames[1:])
        estimates = ['---------------']+list(self.beta)

        # This table will eventually contain more metrics
        table_dic = dict(zip(coef_labels, estimates))

        coef_str = ' + '.join(self.varnames[1:])+'\n'

        print('\nsummary:')
        print('\n{} ~ {}'.format(self.varnames[0], coef_str))
        print('\033[1m'+"{:<15} {:<15}".format('Coefficient','Estimate')+'\033[0m')
        for k, v in sorted(table_dic.items()):
            label = v
            print("{:<15} {:<15}".format(k, label))
        if not self.converged:
            print('\nIRLS has not converged.')
        else:
            print('\nIRLS Converged in {} iterations.'.format(self.converged_at))

        return None

    def predict(self, X, use_prob=False):
        if (not hasattr(self, 'beta')):
            print('Run fit first')
            return None

        pred = X.dot(self.beta)

        if (use_prob):
            odds = np.exp(pred)
            pred = odds/ (1 + odds)

        return pred

