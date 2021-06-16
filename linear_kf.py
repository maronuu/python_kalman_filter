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
    noise_v = np.random.randn(N, 1) * np.sqrt(Q)
    noise_w = np.random.randn(N, 1) * np.sqrt(R)
    x = np.zeros((N, 1))
    y = np.zeros((N, 1))
    y[0] = np.dot(C.T, x[0]) + noise_w[0]
    for k in range(1, N):
        x[k] = np.dot(A, x[k-1]) + B * noise_v[k-1]
        y[k] = np.dot(C.T, x[k]) + noise_w[k]
    
    x_hat = np.zeros((N, 1))
    P = P_0
    x_hat[0] = xhat_0

    for k in range(1, N):
        x_hat[k], P, G = kf(A, B, Bu, C, Q, R, u, y[k], x_hat[k-1], P)
    
    plt.plot(np.arange(0, N), y, c="orange", linestyle="-", label="observation")
    plt.plot(np.arange(0, N), x, c="gray", linestyle="--", label="ground truth")
    plt.plot(np.arange(0, N), x_hat, c="purple", linestyle="-.", label="state estimation")
    plt.xlabel("number of samples")
    plt.legend()
    plt.savefig(f"./image/{figname}")
    plt.show()

def main():
    figname = sys.argv[1]
    # parameters
    A = np.array([1])
    b = np.array([1])
    c = np.array([1])
    Q = np.array([1])
    R = np.array([4])
    xhat_0 = np.array([0])
    P_0 = np.array([0])
    N = 300

    experiment(
        A, b, np.zeros(1), c, Q, R, np.zeros(1), xhat_0, P_0, N, figname
    )

if __name__ == "__main__":
    main()