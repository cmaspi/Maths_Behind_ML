import numpy as np


class PPCA:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.evals = None
        self.evecs = None
        self.sigma_sq = None
        self.W = None

    def fit(self, X: np.ndarray):
        cov = np.dot(X.T, X) / X.shape[0]
        evals, evecs = np.linalg.eigh(cov)
        evals = evals[::-1]
        evecs = evecs[:, ::-1]
        self.evals = evals[:self.n_components]
        self.evecs = evecs[:, :self.n_components]
        self.sigma_sq = evals[self.n_components:].mean()
        diag = np.power(self.evals - self.sigma_sq, 1 / 2)
        diag = np.diag(diag)
        self.W = self.evecs @ diag

    def transform(self, X: np.ndarray) -> np.ndarray:
        M = self.W.T @ self.W
        M += self.sigma_sq * np.eye(M.shape[0])
        M_inv = np.linalg.inv(M)
        z = (M_inv @ self.W.T @ X.T).T
        return z

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def inv_transform(self, Z):
        X = []
        for z in Z:
            mean = self.W @ z
            variance = self.sigma_sq * np.eye(self.W.shape[0])
            x = np.random.multivariate_normal(mean, variance)
            X.append(x)
        return np.array(X)

    def inverse_transform_optimal(self, Z: np.ndarray):
        M = self.W.T @ self.W
        M += self.sigma_sq * np.eye(M.shape[0])
        WW_t_inv = np.linalg.inv(self.W.T @ self.W)
        reconstructed = self.W @ WW_t_inv @ M @ Z.T
        return reconstructed.T

    def gen_data(self, n_samples):
        Z = np.random.multivariate_normal(
            np.zeros(self.n_components),
            np.eye(self.n_components),
            size=n_samples)
        return self.inv_transform(Z)

    def impute_data(self, arr):
        missing_indices = [idx for idx, val in enumerate(arr) if np.isnan(val)]
        w_temp = self.W @ self.W.T + self.sigma_sq * np.eye(self.W.shape[0]) # copying the W matrix
        arr_known = arr[[idx for idx, val in enumerate(arr) if not np.isnan(val)]]

        # row swaps
        pos = 0
        for idx in missing_indices:
            w_temp[idx], w_temp[pos] = w_temp[pos], w_temp[idx]
            pos += 1

        # column swaps
        pos = 0
        for idx in missing_indices:
            w_temp[:, idx], w_temp[:, pos] = w_temp[:, pos], w_temp[:, idx]
            pos += 1

        n_unknown = len(missing_indices)
        sig_22 = w_temp[n_unknown:, n_unknown:]
        sig_12 = w_temp[:n_unknown, n_unknown:]

        arr_missing = sig_12 @ np.linalg.inv(sig_22) @ arr_known
        arr_copy = np.array(arr)
        pos = 0
        for i in range(len(arr_copy)):
            if np.isnan(arr_copy[i]):
                arr_copy[i] = arr_missing[pos]
                pos += 1
        return arr_copy




