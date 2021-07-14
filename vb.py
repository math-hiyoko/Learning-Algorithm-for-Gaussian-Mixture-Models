from time import time

from utils import *


K = 4


def vb(X, K, max_iter=500, eps=1e-2):
    N, Dim = X.shape
    mu, Lambda, pi = params_expect_init(K, Dim)
    alpha, beta, m, nu, W = params_latent_init(K, Dim)
    pi_gauss = get_pi_gauss(X, [mu, Lambda, pi])
    Lambda = nu.reshape(-1, 1, 1) * W
    gamma = get_gamma_latent(X, [mu, Lambda, pi], [alpha, beta, m, nu, W])
    prev_ev = evaluate(pi_gauss)

    for it in range(max_iter):
        S1, Sx, Sxx = get_S(X, gamma)
        beta_0 = beta[:]
        m_0 = m[:]
        alpha = alpha + S1
        beta = beta + S1
        m = (beta_0.reshape(-1, 1) * m + Sx) / beta.reshape(-1, 1)
        nu = nu + S1
        W = np.array([np.linalg.inv(np.linalg.inv(W_k) + beta_0k * np.outer(m_0k, m_0k) \
                                    + Sxx_k - beta_k * np.outer(m_k, m_k))
                      for W_k, beta_0k, beta_k, m_0k, m_k, Sxx_k in zip(W, beta_0, beta, m_0, m, Sxx)])
        pi = alpha / np.sum(alpha)
        mu = m[:]
        Lambda = nu.reshape(-1, 1, 1) * W
        pi_gauss = get_pi_gauss(X, [mu, Lambda, pi])
        gamma = get_gamma_latent(X, [mu, Lambda, pi], [alpha, beta, m, nu, W])
        ev = evaluate(pi_gauss)
        if abs(ev - prev_ev) < eps:
            break
        prev_ev = ev

    return ev, gamma, [mu, Lambda, pi], [alpha, beta, m, nu, W]


def main():
    X = data_input()
    print(f"\nRunning VB algorithm for K = {K}...")
    start = time()
    ev, gamma, params_expect, params_latent = vb(X, K)
    stop = time()
    print(f"Finished in {stop - start:.3f} sec.")
    print(f"log likelihood = {ev:5f}.")

    print("Recording params...")
    expect_data_output(ev, gamma, params_expect)
    latent_data_output(params_latent)
    print("Parameter recorded.\n")
    return


if __name__ == "__main__":
    main()
