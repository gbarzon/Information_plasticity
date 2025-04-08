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
def tmp_factor(w, k, r, D, g1, g2, h1, h2):
    mu1 = theo_mean(w, k, r, g1, h1)
    mu2 = theo_mean(w, k, r, g2, h2)
    sigma1 = theo_sigma(w, k, r, D, g1)
    sigma2 = theo_sigma(w, k, r, D, g2)
    
    return (mu1-mu2).T @ np.linalg.inv(sigma2) @ (mu1 - mu2)

@njit
def kl(w, k, r, D, g1, g2, h1, h2):
    mu1 = theo_mean(w, k, r, g1, h1)
    mu2 = theo_mean(w, k, r, g2, h2)
    sigma1 = theo_sigma(w, k, r, D, g1)
    sigma2 = theo_sigma(w, k, r, D, g2)
    sigma2_inv = np.linalg.inv(sigma2)
    
    sum1 = (mu1-mu2).T @ sigma2_inv @ (mu1 - mu2)
    sum2 = np.trace(sigma2_inv @ sigma1) - np.log(np.linalg.det(sigma1)/np.linalg.det(sigma2)) -2
    
    return 0.5 * (sum1 + sum2)

@njit
def ch(w, k, h1, h2):
    #TODO
    return tmp_factor(w, k, r, D, h1, h2) / 8

def theo_lb(w, k, r, D, pi, gs, hs):
    n_inputs = len(pi)
    
    ress = 0
    for idx_i in range(n_inputs):
        tmp = 0
        for idx_j in range(n_inputs):
            tmp += pi[idx_j] * np.exp( -ch(w, k, r, D, hs[:,idx_i],hs[:,idx_j]) )
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