from time import time

from utils import *


K = 4


def em(X, K, max_iter=500, eps=1e-2):
    N, Dim = X.shape
    mu, Lambda, pi = params_expect_init(K, Dim)
    pi_gauss = get_pi_gauss(X, [mu, Lambda, pi])
    gamma = pi_gauss / np.sum(pi_gauss, axis=1).reshape(-1, 1)
    prev_ev = evaluate(pi_gauss)

    for it in range(max_iter):
        S1, Sx, Sxx = get_S(X, gamma)
        pi = S1 / np.sum(S1)
        mu = Sx / S1.reshape(-1, 1)
        Lambda = list(map(np.linalg.pinv, Sxx / S1.reshape(-1, 1, 1) - [np.outer(mu_k, mu_k) for mu_k in mu]))
        pi_gauss = get_pi_gauss(X, [mu, Lambda, pi])
        gamma = pi_gauss / np.sum(pi_gauss, axis=1).reshape(-1, 1)
        ev = evaluate(pi_gauss)
        if abs(ev - prev_ev) < eps:
            break
        prev_ev = ev

    return ev, gamma, [mu, Lambda, pi]


def main():
    X = data_input()
    print(f"\nRunning EM algorithm for K = {K}...")
    start = time()
    ev, gamma, params = em(X, K)
    stop = time()
    print(f"Finished in {stop-start:.3f} sec.")
    print(f"log likelihood = {ev:5f}.")

    print("Recording params...")
    expect_data_output(ev, gamma, params)
    print("Parameter recorded.\n")
    return


if __name__ == "__main__":
    main()
