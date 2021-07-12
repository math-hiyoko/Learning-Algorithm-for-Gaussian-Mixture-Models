import math
import os
import sys

import numpy as np
from scipy.special import digamma, logsumexp


def data_input():
    args = sys.argv
    argc = len(args)
    input_file = args[1] if argc >= 2 else "x.csv"

    if os.path.isfile(input_file):
        X = np.loadtxt(input_file, delimiter=",")
    else:
        print(f"Error: Input file {input_file} does not exist.")
        sys.exit()

    return X


def expect_data_output(ev, gamma, params_latent):
    args = sys.argv
    argc = len(args)
    z_file = args[2] if argc >= 3 else "z.csv"
    params_file = args[3] if argc >= 4 else "params.dat"

    np.savetxt(z_file, gamma, fmt="%.5f", delimiter=",")

    mu, Lambda, pi = params_latent
    K = len(pi)
    with open(params_file, "w") as f:
        f.write(f"# of cluster: {len(pi):d}\n\n")
        f.write(f"log likelihood {ev:.5f}\n\n")
        for k in range(K):
            f.write(f"cluster {k:d}\npi:\n{pi[k]:.5f}\nmu:\n")
            np.savetxt(f, mu[k], fmt="%.5f", newline=" ")
            f.write("\nLambda:\n")
            np.savetxt(f, Lambda[k], fmt="%.5f")
            f.write("\n")
    return


def latent_data_output(params_latent):
    args = sys.argv
    argc = len(args)
    params_file = args[3] if argc >= 4 else "params.dat"
    alpha, beta, m, nu, W = params_latent
    K = len(alpha)
    with open(params_file, "a") as f:
        f.write("\nLatent Variable\n\n")
        for k in range(K):
            f.write(f"cluster {k:d}\nalpha:\n{alpha[k]:.5e}\nbeta:\n{beta[k]:.5e}\nm:\n")
            np.savetxt(f, m[k], fmt="%.5f", newline=" ")
            f.write(f"\nnu:\n{nu[k]:.5e}\nW:\n")
            np.savetxt(f, W[k], fmt="%.5e")
            f.write("\n")
    return


def params_expect_init(K, Dim):
    mu = np.random.randn(K, Dim)
    Lambda = np.array([np.identity(Dim) for _ in range(K)])
    pi = np.full(K, 1. / K)
    return [mu, Lambda, pi]


def params_latent_init(K, Dim):
    alpha = np.ones(K)
    beta = np.ones(K)
    m = np.random.randn(K, Dim)
    nu = np.full(K, Dim)
    W = np.array([np.identity(Dim) for _ in range(K)])
    return [alpha, beta, m, nu, W]


def get_pi_gauss(X, params):  # [n, k] = pi_k * N(X_n | mu_k, Lambda_k^-1)
    mu, Lambda, pi = params
    N, Dim = X.shape
    det = np.maximum(list(map(np.linalg.det, Lambda)), 0)
    sqrt_det = np.sqrt(det)
    exp = np.exp([[-0.5 * (X_n - mu_k) @ Lambda_k @ (X_n - mu_k).T
                   for mu_k, Lambda_k in zip(mu, Lambda)]
                  for X_n in X])
    pi_gauss = pow(2 * math.pi, -Dim / 2) * pi * sqrt_det * exp
    return pi_gauss


def get_gamma_latent(X, params_expect, params_latent):
    N, Dim = X.shape
    _, Lambda, pi = params_expect
    alpha, beta, m, nu, W = params_latent
    E_log_pi = digamma(alpha) - digamma(np.sum(alpha))
    E_log_Lambda = np.array([np.sum([digamma((nu_k - i) / 2) for i in range(Dim)]) \
                             + Dim * np.log(2.) + np.log(np.linalg.det(W_k))
                             for nu_k, W_k in zip(nu, W)])
    log_rho = E_log_pi + 0.5 * E_log_Lambda - 0.5 * Dim * np.log(2 * np.pi) \
              - 0.5 * np.array([[Dim / beta_k + nu_k * (X_n - m_k) @ W_k @ (X_n - m_k).T
                           for beta_k, m_k, nu_k, W_k in zip(beta, m, nu, W)]
                          for X_n in X])
    log_gamma = log_rho - logsumexp(log_rho, axis=1).reshape(-1, 1)
    gamma = np.exp(log_gamma)
    return gamma


def get_S(X, gamma):  # S_k[1], S_k[x], S_k[xx^T]
    S1 = np.sum(gamma, axis=0)
    Sx = gamma.T @ X
    XXT = np.array([np.outer(X_n, X_n) for X_n in X])
    Sxx = np.sum([gamma_k.reshape(-1, 1, 1) * XXT
                  for gamma_k in gamma.T], axis=1)
    return S1, Sx, Sxx


def evaluate(pi_gauss):  # log likelihood
    log_likelihood = np.sum(np.log(np.sum(pi_gauss, axis=1)))
    return log_likelihood
