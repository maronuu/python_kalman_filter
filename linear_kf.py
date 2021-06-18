from typing import Tuple
import sys
import numpy as np
import matplotlib.pyplot as plt


def kf(
    A : np.ndarray,
    B : np.ndarray,
    Bu : np.ndarray,
    C : np.ndarray,
    Q : np.ndarray,
    R : np.ndarray,
    u : np.ndarray,
    y : np.ndarray,
    xhat : np.ndarray,
    P : np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # reshape to col vector
    xhat = xhat.reshape(-1, 1)
    u = u.reshape(-1, 1)
    y = y.reshape(-1, 1)
    # a preori
    xhatm = np.dot(A, xhat) + np.dot(Bu, u)
    Pm = np.dot(A, np.dot(P, A.T)) + np.dot(B, np.dot(Q, B.T))
    # kalman gain
    G = np.dot(Pm, C) / (np.dot(C.T, np.dot(Pm, C)) + R)
    # a posteriori
    xhat_new = xhatm + np.dot(G, (y - np.dot(C.T, xhatm)))
    P_new = np.dot((np.identity(A.shape[0]) - np.dot(G, C.T)), Pm)

    return (
        xhat_new, P_new, G
    )

def experiment(
    A : np.ndarray,
    B : np.ndarray,
    Bu : np.ndarray,
    C : np.ndarray,
    Q : np.ndarray,
    R : np.ndarray,
    u : np.ndarray,
    # y : np.ndarray,
    xhat_0 : np.ndarray,
    P_0 : np.ndarray,
    N: int,
    figname: str,
) -> np.ndarray:
    dim = A.shape[0]
    noise_v = np.random.randn(N, 1) * np.sqrt(Q)
    noise_w = np.random.randn(N, 1) * np.sqrt(R)
    x = np.zeros((N, dim))
    y = np.zeros((N, 1))
    y[0] = np.dot(C.T, x[0]) + noise_w[0]
    for k in range(1, N):
        x[k] = np.dot(A, x[k-1]) + np.dot(B, noise_v[k-1])
        y[k] = np.dot(C.T, x[k]) + noise_w[k]
    
    x_hat = np.zeros((N, dim))
    P = P_0
    x_hat[0] = xhat_0.reshape(-1)

    for k in range(1, N):
        x_hat_new, P, G = kf(A, B, Bu, C, Q, R, u, y[k], x_hat[k-1], P)
        x_hat[k] = x_hat_new.reshape(-1)
    
    fig, axes = plt.subplots(dim, 1)
    if dim == 1:
        axes.plot(np.arange(0, N), y[:, 0], c="orange", linestyle="-", label="observation")
        axes.plot(np.arange(0, N), x[:, 0], c="gray", linestyle="--", label="ground truth")
        axes.plot(np.arange(0, N), x_hat[:, 0], c="purple", linestyle="-.", label="state estimation")
        axes.legend()
    else:
        for i in range(dim):
            axes[i].plot(np.arange(0, N), y[:, 0], c="orange", linestyle="-", label="observation")
            axes[i].plot(np.arange(0, N), x[:, i], c="gray", linestyle="--", label="ground truth")
            axes[i].plot(np.arange(0, N), x_hat[:, i], c="purple", linestyle="-.", label="state estimation")
            axes[i].legend()

    plt.savefig(figname)
    plt.show()

def main():
    figname = sys.argv[1]
    # parameters
    A = np.array([
        [1.0]
    ])
    b = np.array([
        [1.5,],
    ])
    c = np.array([
        [0.36,],
    ])
    Q = 100.0
    R = 1.0
    xhat_0 = np.array([
        [0.0,],
    ])
    gamma = 0.6879987098
    P_0 = np.identity(1) * gamma
    N = 50

    experiment(
        A, b, np.zeros(1), c, Q, R, np.zeros(1), xhat_0, P_0, N, figname
    )

if __name__ == "__main__":
    main()