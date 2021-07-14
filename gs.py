from time import time

from scipy.stats import wishart

from utils import *


K = 4


def get_z(gamma):
    N, K = gamma.shape
    identity = np.identity(K)
    Z = np.array([identity[np.random.choice(list(range(K)), p=g)] for g in gamma])
    return Z


def gs(X, K, max_iter=500, eps=1e-2):
    N, Dim = X.shape
    mu, Lambda, pi = params_expect_init(K, Dim)
    alpha, beta, m, nu, W = params_latent_init(K, Dim)
    pi_gauss = get_pi_gauss(X, [mu, Lambda, pi])
    Lambda = nu.reshape(-1, 1, 1) * W
    gamma = get_gamma_latent(X, [mu, Lambda, pi], [alpha, beta, m, nu, W])
    prev_ev = evaluate(pi_gauss)

    for it in range(max_iter):
        Z = get_z(gamma)
        S1, Sx, Sxx = get_S(X, Z)
        beta_0 = beta[:]
        m_0 = m[:]
        alpha = alpha + S1
        beta = beta + S1
        m = (beta_0.reshape(-1, 1) * m + Sx) / beta.reshape(-1, 1)
        nu = nu + S1
        W = np.array([np.linalg.inv(np.linalg.inv(W_k) + beta_0k * np.outer(m_0k, m_0k) \
                                    + Sxx_k - beta_k * np.outer(m_k, m_k))
                      for W_k, beta_0k, beta_k, m_0k, m_k, Sxx_k in zip(W, beta_0, beta, m_0, m, Sxx)])
        Lambda = np.array([wishart(df=nu_k, scale=W_k).rvs(1) for nu_k, W_k in zip(nu, W)])
        pi = np.random.dirichlet(alpha, 1)[0]
        mu = np.array([np.random.multivariate_normal(m_k, np.linalg.inv(beta_k * Lambda_k), 1)[0]
                       for m_k, beta_k, Lambda_k in zip(m, beta, Lambda)])
        pi_gauss = get_pi_gauss(X, [mu, Lambda, pi])
        gamma = get_gamma_latent(X, [mu, Lambda, pi], [alpha, beta, m, nu, W])
        ev = evaluate(pi_gauss)
        if abs(ev - prev_ev) < eps:
            break
        prev_ev = ev

    Z = get_z(gamma)
    return ev, Z, [mu, Lambda, pi], [alpha, beta, m, nu, W]


def main():
    X = data_input()
    print(f"\nRunning GS algorithm for K = {K}...")
    start = time()
    ev, Z, params_expect, params_latent = gs(X, K)
    stop = time()
    print(f"Finished in {stop - start:.3f} sec.")
    print(f"log likelihood = {ev:5f}.")

    print("Recording params...")
    expect_data_output(ev, Z, params_expect)
    latent_data_output(params_latent)
    print("Parameter recorded.\n")
    return


if __name__ == "__main__":
    main()
