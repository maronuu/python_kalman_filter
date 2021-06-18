from typing import Callable, Tuple
import sys
import numpy as np
import matplotlib.pyplot as plt


def ukf(
    f: Callable[[np.ndarray], np.ndarray],
    h: Callable[[np.ndarray], np.ndarray],
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    y: np.ndarray,
    xhat: np.ndarray,
    P: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # reshape to col vector
    xhat = xhat.reshape(-1, 1)
    y = y.reshape(-1, 1)
    # a preori
    xhatm, Pm, _ = u_transform(f, xhat, P)
    Pm = Pm + B.dot(Q.dot(B.T))
    yhatm, Pyy, Pxy = u_transform(h, xhatm, Pm)
    # kalman gain
    G = Pxy / (Pyy + R)
    # a preori
    xhat_new = xhatm + G * (y - yhatm)
    P_new = Pm - G * Pxy.T

    return xhat_new, P_new, G

def u_transform(
    f: Callable[[np.ndarray], np.ndarray],
    xm: np.ndarray,
    Pxx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    xm = xm.reshape(-1, 1)
    n_dim = xm.shape[0]
    kappa = 3 - n_dim
    w0 = kappa / (n_dim + kappa)
    wi = 1 / (2 * (n_dim + kappa))
    w_diag = np.array([w0] + [wi] * (2*n_dim))
    W = np.diag(w_diag)

    L = np.linalg.cholesky(Pxx)
    X = np.concatenate((
        xm,
        np.ones((n_dim, 1)).dot(xm.T) + np.sqrt(n_dim + kappa) * L,
        np.ones((n_dim, 1)).dot(xm.T) - np.sqrt(n_dim + kappa) * L,
    ))
    Y = np.apply_along_axis(f, 0, X)
    ym = np.sum(W.dot(Y)).T
    Pyy = (Y - ym).T.dot(W.dot((Y - ym)))
    Pxy = (X - xm).T.dot(W.dot((Y - ym)))

    return ym , Pyy, Pxy

def experiment(
    dim: int,
    f: Callable[[np.ndarray], np.ndarray],
    h: Callable[[np.ndarray], np.ndarray],
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    # y: np.ndarray,
    xhat_0: np.ndarray,
    P_0: np.ndarray,
    N: int,
    figname: str,
) -> np.ndarray:
    v = np.random.randn(N, 1) * np.sqrt(Q)
    w = np.random.randn(N, 1) * np.sqrt(R)
    x = np.zeros((N, dim))
    y = np.zeros((N, 1))
    x[0] = xhat_0
    y[0] = h(x[0])
    for k in range(1, N):
        x[k] = f(x[k-1]) + B.dot(v[k-1])
        y[k] = h(x[k]) + w[k]

    xhat = np.zeros((N, dim))
    P = P_0
    xhat[0] = xhat_0.reshape(-1)
    for k in range(1, N):
        xhat_new, P, G = ukf(
            f, h, B, Q, R, y[k], xhat[k-1], P,
        )
        xhat[k] = xhat_new.reshape(-1)
    
    fig, axes = plt.subplots(dim, 1)
    if dim == 1:
        axes.plot(np.arange(0, N), y[:, 0], c="orange", linestyle="-", label="observation")
        axes.plot(np.arange(0, N), x[:, 0], c="gray", linestyle="--", label="ground truth")
        axes.plot(np.arange(0, N), xhat[:, 0], c="purple", linestyle="-.", label="state estimation")
        axes.legend()
    else:
        for i in range(dim):
            axes[i].plot(np.arange(0, N), y[:, 0], c="orange", linestyle="-", label="observation")
            axes[i].plot(np.arange(0, N), x[:, i], c="gray", linestyle="--", label="ground truth")
            axes[i].plot(np.arange(0, N), xhat[:, i], c="purple", linestyle="-.", label="state estimation")
            axes[i].legend()

    plt.savefig(figname)
    plt.show()


def main():
    figname = sys.argv[1]
    # parametrers
    f = lambda x: 0.2*x + 25*x/(1+x**2) + 10*np.cos(x/10) + 0.01 * np.exp(-x)
    h = lambda x: 1/20*(x**2)
    b = np.array([[1.0]])
    N = 50
    Q = np.array([[1.0]])
    R = np.array([[3.0]])
    xhat_0 = np.array([[0.0]])
    gamma = 10.0
    P_0 = gamma * np.identity(1)
    experiment(
        1, f, h, b, Q, R, xhat_0, P_0, N, figname
    )

if __name__ == "__main__":
    main()