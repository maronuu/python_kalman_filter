from typing import Callable, Tuple
import sys
import numpy as np
import matplotlib.pyplot as plt


def ekf(
    f: Callable[[np.ndarray], np.ndarray],
    h: Callable[[np.ndarray], np.ndarray],
    A: Callable[[np.ndarray], np.ndarray],
    B: np.ndarray,
    C: Callable[[np.ndarray], np.ndarray],
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
    xhatm = f(xhat)
    Pm = A(xhat).dot(P.dot(A(xhat).T)) + B.dot(Q.dot(B.T))
    # kalman gain
    G = Pm.dot(C(xhatm)) / (C(xhatm).T.dot(Pm.dot(C(xhatm))) + R)
    # a posteriori
    xhat_new = xhatm + G.dot(y - h(xhatm))
    P_new = (np.identity(A(xhat).shape[0]) - G.dot(C(xhatm).T)).dot(Pm)

    return xhat_new, P_new, G


def experiment(
    dim: int,
    f: Callable[[np.ndarray], np.ndarray],
    h: Callable[[np.ndarray], np.ndarray],
    A: Callable[[np.ndarray], np.ndarray],
    B: np.ndarray,
    C: Callable[[np.ndarray], np.ndarray],
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
        xhat_new, P, G = ekf(
            f, h, A, B, C, Q, R, y[k], xhat[k-1], P
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
    a = lambda x: 0.2 + (25 - 25*x**2 - 25*x**3) / ((1+x**2)**2) - np.sin(x/10) - 0.01*np.exp(-x)
    b = np.array([[1.0]])
    c = lambda x: 1/10 * x
    N = 50
    Q = np.array([[1.0]])
    R = np.array([[3.0]])
    xhat_0 = np.array([[0.0]])
    gamma = 10.0
    P_0 = gamma * np.identity(1)
    experiment(
        1, f, h, a, b, c, Q, R, xhat_0, P_0, N, figname
    )

if __name__ == "__main__":
    main()