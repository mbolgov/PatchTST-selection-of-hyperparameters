import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def build_embedding(X, tau, m_max):
    """
    Строит delay embedding для всех размерностей до m_max
    """
    T, d = X.shape
    length = T - (m_max - 1) * tau
    Y = np.zeros((length, m_max * d))
    for i in range(m_max):
        Y[:, i * d:(i + 1) * d] = X[i * tau:i * tau + length]
    return Y


def plot_delay_embedding(X, tau=8, m=30, data_name="", left=0, right=10000):
    """
    Визуализирует delay embedding (PC1 vs PC2).
    """
    Y = build_embedding(X, tau, m)
    reducer = PCA(n_components=2)
    Y_red = reducer.fit_transform(Y)

    plt.figure(figsize=(10, 5))
    plt.plot(Y_red[left:right, 0], Y_red[left:right, 1], color="navy", lw=1)
    plt.xlabel("PC1", fontsize=16)
    plt.ylabel("PC2", fontsize=16)
    plt.title(f"Delay Embedding{" for " + data_name if data_name else ""} (tau={tau}, m={m})", fontsize=20)
    plt.grid(linestyle=":")
    plt.tight_layout()
    plt.show()
