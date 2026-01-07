import numpy as np

def update_cond_mean(X,mean,kernel):
    # for a given data sample and hypothesized mean, calculate the conditional deviance on dependent spot set
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
    # for a given data sample and hypothesized mean and variance,  compute the log likelihood
    N,G = X.shape
    if isinstance(delta,int):
        delta = np.array([delta])
    if isinstance(sigma_sq,int):
        sigma_sq = np.array([sigma_sq])
    K = len(delta)
    ll = np.zeros((G,K))

    cond_dev = update_cond_mean(X,mean,kernel)
    cond_cov_eig = update_cond_cov(kernel,delta)
    
    for k in range(K):
        ll[:,k] = np.log(2 * np.pi)*N + 2*np.log(sigma_sq[k])*N
        for i in range(kernel.M):
            det = np.prod(cond_cov_eig[k][i][0])
            if det <= 0:
                print(cond_cov_eig[k][i][0]) 
            ll[:,k] += np.log(det)
            temp = cond_dev[k][kernel.ss_loc[i],:].T @ cond_cov_eig[k][i][1]
            ll[:,k] += np.sum(np.multiply(1/cond_cov_eig[k][i][0],np.square(temp)),axis=1)/sigma_sq[k]
    ll = ll*-0.5
    return ll

def LikRatio_Test(X,kernel_1,kernel_2,mean_1,mean_2,sigma_sq_1,sigma_sq_2,delta_1,delta_2):
    ll_1 = GaussianNLL(kernel_1,mean_1,sigma_sq_1,delta_1)
    ll_2 = GaussianNLL(kernel_2,mean_2,sigma_sq_2,delta_2)
    lr_stat = 2 * (ll_2 - ll_1)
    if ll_1 > ll_2:
        print("Model 1 fits better.")
    elif ll_2 > ll_1:
        print("Model 2 fits better.")