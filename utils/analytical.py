import numpy as np
from numba import njit

@njit
def theo_mean(w, k, r, g, h):
    ### Compute expected average
    beta = r * (r + g*w*(k-1))

    xplus = (h[0]*r + g*w*k*(h[0]-h[1])) / beta
    xminus = (h[1]*(r-g*w) + g*w*h[0]) / beta
    
    return np.array( [xplus, xminus] )

@njit
def theo_sigma(w, k, r, D, g):
    ### Compute expected covariance matrix
    tt = r*(r+g*w*(k-1))*(2*r+g*w*(k-1))
    
    sigma = np.array([[2*r**2+(-1+3*k)*g*w*r+2*g**2*k**2*w**2, g*w*(r-k*r+2*g*k*w)],
                      [g*w*(r-k*r+2*g*k*w), 2*r**2+(k-3)*g*r*w+2*g**2*w**2]])
    
    return D * sigma / tt

@njit
def theo_mean_multiple(w, k, r, gs, h):
    n = np.size(gs)
    means = np.zeros((n, 2, 2))
    
    for idx in range(n):
        means[idx] = theo_mean(w, k, r, gs[idx], h)
    
    return means

@njit
def theo_sigma_multiple(w, k, r, D, gs):
    n = np.size(gs)
    sigmas = np.zeros((n, 2, 2))
    
    for idx in range(n):
        sigmas[idx] = theo_sigma(w, k, r, D, gs[idx])
    
    return sigmas

@njit
def kl(w, k, r, D, g1, g2, h1, h2):
    ### Compute the KL divergence between two Gaussian distributions
    mu1 = theo_mean(w, k, r, g1, h1)
    mu2 = theo_mean(w, k, r, g2, h2)
    sigma1 = theo_sigma(w, k, r, D, g1)
    sigma2 = theo_sigma(w, k, r, D, g2)
    sigma2_inv = np.linalg.inv(sigma2)
    
    sum1 = (mu1-mu2).T @ sigma2_inv @ (mu1 - mu2)
    sum2 = np.trace(sigma2_inv @ sigma1) - np.log(np.linalg.det(sigma1)/np.linalg.det(sigma2)) - sigma1.shape[0]
    
    return 0.5 * (sum1 + sum2)

@njit
def ch(w, k, r, D, g1, g2, h1, h2, alpha=0.5):
    ### Compute the Chernoff divergence between two Gaussian distributions
    mu1 = theo_mean(w, k, r, g1, h1)
    mu2 = theo_mean(w, k, r, g2, h2)
    sigma1 = theo_sigma(w, k, r, D, g1)
    sigma2 = theo_sigma(w, k, r, D, g2)
    
    sum1 = (1-alpha) * alpha * (mu1-mu2).T @ np.linalg.inv((1-alpha)*sigma1 + alpha*sigma2) @ (mu1 - mu2)
    sum2 = np.log( np.linalg.det((1-alpha)*sigma1 + alpha*sigma2) / np.linalg.det(sigma1)**(1-alpha) / np.linalg.det(sigma2)**alpha )

    return 0.5 * (sum1 + sum2)

def theo_lb(w, k, r, D, pi, gs, hs):
    n_inputs = len(pi)
    
    ress = 0
    for idx_i in range(n_inputs):
        tmp = 0
        for idx_j in range(n_inputs):
            tmp += pi[idx_j] * np.exp( -ch(w, k, r, D, gs[idx_i], gs[idx_j], hs[:,idx_i], hs[:,idx_j]) )
        ress += pi[idx_i] * np.log( tmp )
    return - ress

def theo_ub(w, k, r, D, pi, gs, hs):
    n_inputs = len(pi)

    ress = 0
    for idx_i in range(n_inputs):
        tmp = 0
        for idx_j in range(n_inputs):
            tmp += pi[idx_j] * np.exp( -kl(w, k, r, D, gs[idx_i], gs[idx_j], hs[:,idx_i], hs[:,idx_j]) )
        ress += pi[idx_i] * np.log( tmp )
    return - ress