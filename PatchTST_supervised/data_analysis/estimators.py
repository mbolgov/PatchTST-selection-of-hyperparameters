import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed

from delay_embedding import build_embedding


class DelayEstimator:
    def __init__(self, max_tau=100, bins=20, data_name="Time Series", n_jobs=-1):
        self.max_tau = max_tau
        self.bins = bins
        self.data_name = data_name
        self.n_jobs = n_jobs
        self.ami = None
        self.tau_min = None
        self.tau_max = None

    def _ami_for_tau(self, X, tau):
        """
        Вычисляет AMI для многомерного ряда, усреднённое по признакам для фиксированного tau
        """
        n, d = X.shape
        edges = [np.histogram(X[:, i], bins=self.bins)[1] for i in range(d)]
        X_digitized = np.column_stack([np.digitize(X[:, i], edges[i]) for i in range(d)])
        mi_vals = [mutual_info_score(X_digitized[:-tau, i], X_digitized[tau:, i]) for i in range(d)]
        return np.mean(mi_vals)

    @staticmethod
    def _find_first_local_minimum(values):
        """
        Возвращает индекс первого локального минимума массива или None
        """
        # смотрим на окно длины 9
        k = 4
        center = values[k:len(values) - k]
        is_local_minimum = np.ones_like(center, dtype=bool)

        for shift in range(-k, k + 1):
            is_local_minimum &= (center <= values[k + shift:len(values) - k + shift])
        ids = np.flatnonzero(is_local_minimum) + k + 1
        return ids[0] if ids.size else None

    @staticmethod
    def _find_first_local_maximum(values):
        """
        Возвращает индекс первого локального максимума массива или None
        """
        # смотрим на окно длины 9
        k = 4
        center = values[k:len(values) - k]
        is_local_minimum = np.ones_like(center, dtype=bool)

        for shift in range(-k, k + 1):
            is_local_minimum &= (center >= values[k + shift:len(values) - k + shift])
        ids = np.flatnonzero(is_local_minimum) + k + 1
        return ids[0] if ids.size else None

    def fit(self, X):
        """
        Вычисляет AMI для многомерного ряда, усреднённое по признакам, находит первые экстремумы
        """
        self.ami = Parallel(n_jobs=self.n_jobs)(
            delayed(self._ami_for_tau)(X, tau) for tau in range(1, self.max_tau + 1)
        )
        self.ami = np.array(self.ami)

        self.tau_min = self._find_first_local_minimum(self.ami)
        self.tau_max = self._find_first_local_maximum(self.ami)

        return self

    def plot(self):
        """
        Строит график AMI, отмечает первый локальный минимум и максимум
        """
        if self.ami is None:
            raise RuntimeError("Run .fit(X) first")

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(self.ami) + 1), self.ami, marker="o", linestyle="-", color="navy", label="AMI")

        # отображаем локальный минимум
        if self.tau_min is not None:
            plt.axvline(self.tau_min, color="red", linestyle="--", lw=2, label=f"first min τ = {self.tau_min}")
            plt.scatter([self.tau_min], [self.ami[self.tau_min - 1]], color="red", zorder=5, s=80, edgecolor="k")

        # отображаем локальный максимум
        if self.tau_max is not None:
            plt.axvline(self.tau_max, color="green", linestyle="--", lw=2, label=f"first max τ = {self.tau_max}")
            plt.scatter([self.tau_max], [self.ami[self.tau_max - 1]], color="green", zorder=5, s=80, edgecolor="k")

        plt.title(f"Average Mutual Information — {self.data_name}", fontsize=20)
        plt.xlabel("Lag (τ)", fontsize=16)
        plt.ylabel("AMI", fontsize=16)
        plt.grid(linestyle=":")
        plt.legend(fontsize=14)
        plt.tight_layout()
        plt.show()


class EmbeddingDimensionEstimator:
    def __init__(self, tau, m_max=100, data_name="", Rtol=10.0, Atol=2.0, threshold=0.01, multivariate=True, n_jobs=-1):
        self.tau = tau
        self.m_max = m_max
        self.data_name = data_name
        self.Rtol = Rtol
        self.Atol = Atol
        self.threshold = threshold
        self.multivariate = multivariate
        self.n_jobs = n_jobs
        self.all_fnns = None
        self.fnn_max = None
        self.coverage_ratio = None
        self.m_opt = None

    def _fnn_for_m(self, Y_full, X, m):
        """
        Вычисляет долю FNN для фиксированного m
        """
        T, d = X.shape
        N = T - m * self.tau
        Y_m1 = Y_full[:N, :(m + 1) * d]
        Y_m = Y_full[:N, :m * d]

        neighbors = NearestNeighbors(n_neighbors=2).fit(Y_m)
        dist, idx = neighbors.kneighbors(Y_m)
        dist, idx = dist[:, 1], idx[:, 1]

        dist_m1 = np.linalg.norm(Y_m1 - Y_m1[idx], axis=1)
        R = np.sqrt(np.maximum(dist_m1 ** 2 - dist ** 2, 0)) / (dist + 1e-8)
        A = np.linalg.norm((Y_m1[:, -d:] - Y_m1[idx, -d:]) / np.std(X, axis=0), axis=1)
        return np.sum((R > self.Rtol) | (A > self.Atol)) / N

    def _fnn_ratio_parralel(self, X):
        """
        Вычисляет долю FNN для размерностей от 1 до m_max параллельно
        """
        Y_full = build_embedding(X, tau=self.tau, m_max=self.m_max)
        fnn_percents = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fnn_for_m)(Y_full, X, m) for m in range(1, self.m_max)
        )
        return np.array(fnn_percents)

    def _fnn_ratio(self, X):
        """
        Вычисляет долю FNN для размерностей от 1 до m_max последовательно
        """
        Y_full = build_embedding(X, tau=self.tau, m_max=self.m_max)
        fnn_percents = [self._fnn_for_m(Y_full, X, m) for m in range(1, self.m_max)]
        return np.array(fnn_percents)

    def filter_fnns(self, threshold=None):
        """
        Фильтрует FNN для каждого канала и находит оптимальную размерность
        """
        if self.all_fnns is None:
            raise RuntimeError("Run .fit(X) first")

        if threshold is not None:
            self.threshold = threshold
        valid_fnns = [f for f in self.all_fnns if f[-1] < self.threshold] if not self.multivariate else self.all_fnns
        self.fnn_max = np.max(valid_fnns, axis=0)
        self.coverage_ratio = len(valid_fnns) / len(self.all_fnns)

        rolling_max = np.array([max(self.fnn_max[i:i + 5]) for i in range(self.m_max - 1)])
        below = np.where(rolling_max < self.threshold)[0]
        self.m_opt = int(below[0]) + 1 if len(below) > 0 else None
        return self

    def fit(self, X):
        """
        Вычисляет FNN для каждого канала и находит оптимальную размерность
        """
        T, d = X.shape
        if self.multivariate:
            self.all_fnns = [self._fnn_ratio_parralel(X)]
        elif d < self.m_max:
            self.all_fnns = [self._fnn_ratio_parralel(X[:, [j]]) for j in range(d)]
        else:
            self.all_fnns = Parallel(n_jobs=self.n_jobs)(
                delayed(self._fnn_ratio)(X[:, [j]]) for j in range(d)
            )
        self.filter_fnns()
        return self

    def plot(self):
        """
        Рисует график fnn_max и отмечает оптимальную размерность
        """
        if self.all_fnns is None:
            raise RuntimeError("Run .fit(X) first")

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(self.fnn_max) + 1), self.fnn_max, marker="o", linestyle="-", color="navy", label="max FNN across components")

        # отмечаем оптимальную размерность
        if self.m_opt is not None:
            plt.axvline(self.m_opt, color="red", linestyle="--", lw=2, label=f"m = {self.m_opt}")
            plt.scatter([self.m_opt], [self.fnn_max[self.m_opt - 1]], color="red", zorder=5, s=80, edgecolor="k")

        # изображаем линию порога
        plt.axhline(self.threshold, color="green", linestyle=":", lw=2, label=f"threshold = {self.threshold}")

        # добавляем в легенду, какая часть каналов удовлетворяет выбранному dimension
        plt.plot([], [], " ", label=f"channels covered: {100 * self.coverage_ratio:.1f}%")

        plt.title(f"FNN — Embedding Dimension for {self.data_name}" + ("" if self.multivariate else " (channel-wise max)"), fontsize=20)
        plt.xlabel("Embedding dimension (m)", fontsize=16)
        plt.ylabel("FNN ratio" + ("" if self.multivariate else " (max over channels)"), fontsize=16)
        plt.yscale("log")
        plt.grid(linestyle=":")
        plt.legend(fontsize=14)
        plt.tight_layout()
        plt.show()
