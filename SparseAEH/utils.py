import numpy as np
from .covariance import block_statistics

def update_cond_mean(X,mean,kernel,delta):
    # for a given data sample and hypothesized mean, calculate the conditional deviance on dependent spot set
    K = mean.shape[-1]
    cond_dev = X[np.newaxis,:] - mean.transpose()[:,:,np.newaxis]   #K,N,G
    dev = cond_dev.copy()
    for i in range(kernel.M):
        if len(kernel.dependency[i]) > 0:
            for k in range(K):
                cond_dev[k,kernel.ss_loc[i],:] = cond_dev[k,kernel.ss_loc[i],:] - \
                np.multiply(1/(kernel.ds_eig[i][0]+delta[k]),kernel.A[i]) @ kernel.ds_eig[i][1].T @ dev[k,kernel.ds_loc[i],:]
    return cond_dev

def update_cond_cov(kernel,Delta):
    if isinstance(Delta,int):
        Delta = np.array([Delta])
    cond_cov_eig = [[] for _ in range(len(Delta))] # k clusters
    for eig, delta in zip(cond_cov_eig,Delta):
        for i in range(kernel.M):
            if len(kernel.dependency[i]) == 0:
            #kernel.cond_cov.append(kernel.kernel.base_cond_cov[i]+kernel.delta*np.eye(len(kernel.kernel.ss_loc[i])))
                s,u = np.linalg.eigh(kernel.cond_cov[i]+delta*np.eye(len(kernel.ss_loc[i])))
            else:
                s,u = np.linalg.eigh(kernel.cond_cov[i]+delta*np.eye(len(kernel.ss_loc[i]))+
                                        delta*np.multiply(1/((kernel.ds_eig[i][0]+delta)*kernel.ds_eig[i][0]),kernel.A[i])@kernel.A[i].T)
            eig.append((s,u))
    return cond_cov_eig

def GaussianNLL(X,kernel,mean,sigma_sq,delta):
    """Log densities of each feature under each block-conditional component."""
    X = np.asarray(X, dtype=float)
    mean = np.asarray(mean, dtype=float)
    sigma_sq = np.atleast_1d(np.asarray(sigma_sq, dtype=float))
    delta = np.atleast_1d(np.asarray(delta, dtype=float))
    N, G = X.shape
    K = len(delta)
    if mean.shape != (N, K) or sigma_sq.shape != (K,):
        raise ValueError("mean, sigma_sq, and delta have incompatible shapes")
    if np.any(sigma_sq <= 0) or not np.all(np.isfinite(sigma_sq)):
        raise ValueError("sigma_sq must be positive and finite")
    ll = np.empty((G, K), dtype=float)
    for k in range(K):
        logdet, quadratic = block_statistics(X - mean[:, k, None], kernel, delta[k])
        ll[:, k] = -0.5 * (N * np.log(2 * np.pi * sigma_sq[k])
                           + logdet + quadratic / sigma_sq[k])
    return ll

def LikRatio_Test(X,kernel_1,kernel_2,mean_1,mean_2,sigma_sq_1,sigma_sq_2,delta_1,delta_2):
    ll_1 = GaussianNLL(kernel_1,mean_1,sigma_sq_1,delta_1)
    ll_2 = GaussianNLL(kernel_2,mean_2,sigma_sq_2,delta_2)
    lr_stat = 2 * (ll_2 - ll_1)
    if ll_1 > ll_2:
        print("Model 1 fits better.")
    elif ll_2 > ll_1:
        print("Model 2 fits better.")
