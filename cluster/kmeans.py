import numpy as np
from scipy.spatial.distance import cdist

class KMeans:
    def __init__(self, X, K):
        self.X = X
        self.K = K

        # init centers
        self.centers = [X[np.random.choice(X.shape[0], K, replace=False)]]
        self.labels = []

    def assign_labels(self, centers):
        D = cdist(self.X, centers)
        return np.argmin(D, axis=1)

    def next_centers(self, labels):
        new_centers = np.zeros((self.K, self.X.shape[1]))
        for k in range(self.K):
            Xk = self.X[labels == k, :]
            new_centers[k, :] = np.mean(Xk, axis = 0)

        return new_centers

    def has_converged_with(self, centers):
        prev_centers = set([tuple(a) for a in self.centers[-1]])
        current_centers = set([tuple(a) for a in centers])
        return prev_centers == current_centers

    def cluster(self):
        i = 0
        while True:
            next_labels = self.assign_labels(self.centers[-1])
            self.labels.append(next_labels)
            new_centers = self.next_centers(self.labels[-1])

            if (self.has_converged_with(new_centers)):
                break
                
            self.centers.append(new_centers)
            i += 1

        return (self.centers, self.labels, i)
