import numpy as np
#from function import RBF_kernel
import scipy
import time
from scipy.special import logsumexp
from sklearn.metrics.pairwise import laplacian_kernel,rbf_kernel
from scipy.sparse import csr_matrix
from operator import itemgetter
import warnings
from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix
import random
from scipy.stats import chi2
from .utils import update_cond_mean, update_cond_cov, GaussianNLL, LikRatio_Test
from .covariance import line_search_covariance

def _pinv_1d(v, eps=1e-5):
    return np.array([0 if abs(x) <= eps else 1/x for x in v], dtype=float)

def _pexp(array):
    return np.array([np.inf if n>50 else (np.exp(n) if n >-50 else 0) for n in array])


class Kernel:

    def __init__(self,spatial,ss_loc=None,group_size=16,cov=None,dependency=None,d=5,kernel='laplacian',l=0.01):
        """
        number of superspots: M
        cov: full pre-determined covariance matrix
        dependency format: M-length list of lists, the i-th element indicates the superspots that the i-th superspot is dependent on
        ss_loc: M-length list of lists, the i-th element indicates the spots (ordinal) that the i-th superspots contain
        spatial: spatial coordinates of all spots
        l: hyperparameter for kernel
        """
        super().__init__()
        self.N = len(spatial)
        #self.full_cov = [[0 for _ in range(self.M)] for _ in range(self.M)]
        #self.ss_loc = ss_loc
        self.spatial = spatial
        self.cond_mean = None
        self.l = l
        self.d = d
        self.A = []
        self.cond_cov = []
        self.kernel = kernel
        self._initialize(cov,ss_loc,dependency,group_size)
    
    def get_mat(self,rows,cols):
        row = []
        for r in rows:
            col = []
            for c in cols:
                if r >= c:
                    col.append(self.full_cov[r][c])
                else:
                    col.append(self.full_cov[c][r].T)
            row.append(np.hstack(col))
        return np.vstack(row)

    
    def _initialize(self,cov,ss_loc,dependency,group_size):
        self._init_ss(ss_loc,dependency,group_size)
        self._init_ds_loc()
        self._init_full_cov(cov)
        self._init_ds_eig()
        self._init_base_cond_cov()        
        #self._init_cond_cov()

    def _init_ss(self,ss_loc,dependency,group_size):
        if ss_loc is not None:
            self.ss_loc = ss_loc
            self.dependency = dependency
            self.M = len(dependency)
        else:
            centers = []
            self.ss_loc = []
            group = self.N / group_size
            g_r = max(1,int(np.sqrt(group)))
            g_c = max(1,int(group/g_r))
            #self.M = g_c * g_r
            kmeans_r = KMeans(n_clusters=g_r, random_state=0, n_init=10)
            kmeans_c = KMeans(n_clusters=g_c, random_state=0, n_init=10)
            kmeans_r.fit(self.spatial[:,0:1])
            kmeans_c.fit(self.spatial[:,1:])
            for r in range(g_r):
                for c in range(g_c):
                    pos = np.logical_and(kmeans_r.labels_ == r, kmeans_c.labels_ == c)
                    if pos.any():
                        centers.append(np.average(self.spatial[pos],axis=0))
                        self.ss_loc.append(np.arange(len(self.spatial))[pos])
            #print(self.ss_loc)
            #print(centers)
            centers = np.vstack(centers)
            self.M = centers.shape[0]
            self.dependency = [[] for _ in range(self.M)]
            distance = distance_matrix(centers,centers)
            for i in range(self.M):
                if i <= self.d:
                    self.dependency[i] = list(range(i))
                else:
                    #self.dependency[i] = np.argmax(distance[i,:i],self.d)
                    self.dependency[i] = distance[i,:i].argsort()[:self.d]


    def _init_ds_loc(self):
        self.ds_loc = []        
        for i,ds in enumerate(self.dependency):
           ind = []
           for ss in ds:
               ind += np.array(self.ss_loc[ss]).tolist()
           self.ds_loc.append(ind)
    
    def _init_full_cov(self,cov):
        self.full_cov = [{} for _ in range(self.M)]
        for i in range(self.M):
            if cov is not None:
                self.full_cov[i][i] = cov[np.ix_(self.ss_loc[i],self.ss_loc[i])]
            else:
                if self.kernel == 'rbf':
                    self.full_cov[i][i] = rbf_kernel(self.spatial[self.ss_loc[i]],gamma=self.l)
                else:
                    self.full_cov[i][i] = laplacian_kernel(self.spatial[self.ss_loc[i]],gamma=self.l)
            for j in self.dependency[i]:
                if cov is not None:
                    self.full_cov[i][j] = cov[np.ix_(self.ss_loc[i],self.ss_loc[j])]
                else:
                    if self.kernel == 'rbf':
                        self.full_cov[i][j] = rbf_kernel(self.spatial[self.ss_loc[i]],
                                                        self.spatial[self.ss_loc[j]],gamma=self.l)
                    else:
                        self.full_cov[i][j] = laplacian_kernel(self.spatial[self.ss_loc[i]],
                                                              self.spatial[self.ss_loc[j]],gamma=self.l)
        for k in range(self.M):
            for i in self.dependency[k]:
                for j in self.dependency[k]:
                    if j<i and j not in self.full_cov[i]:
                        if cov is not None:
                            self.full_cov[i][j] = cov[np.ix_(self.ss_loc[i],self.ss_loc[j])]
                        else:
                            if self.kernel == 'rbf':
                                self.full_cov[i][j] = rbf_kernel(self.spatial[self.ss_loc[i]],
                                                                self.spatial[self.ss_loc[j]],gamma=self.l)
                            else:
                                self.full_cov[i][j] = laplacian_kernel(self.spatial[self.ss_loc[i]],
                                                                      self.spatial[self.ss_loc[j]],gamma=self.l)

    def _init_ds_eig(self):
        #C_m,C_m
        self.ds_eig = []
        for i in range(self.M):
            if len(self.dependency[i]) == 0:
                self.ds_eig.append(())
                self.A.append(())            
            else:            
                ds_cov = self.get_mat(self.dependency[i],self.dependency[i])
                s,u = np.linalg.eigh(ds_cov)
                #s_inv = _pinv_1d(s)
                self.ds_eig.append((s,u))
                self.A.append(self.get_mat([i],self.dependency[i]) @ u)
    

    def _init_base_cond_cov(self):
        # sigma when delta=0
        for i in range(self.M):
            if len(self.dependency[i]) == 0:
                self.cond_cov.append(self.get_mat([i],[i]))
            else:
                inverse = _pinv_1d(self.ds_eig[i][0])
                self.cond_cov.append(self.get_mat([i],[i]) - np.multiply(inverse,self.A[i])@self.A[i].T)
    


class MixedGaussian:
    def __init__(self,spatial,ss_loc=None,group_size=16,cov=None,dependency=None,d=5,kernel='rbf',l=0.01):
        self.kernel = Kernel(spatial,ss_loc,group_size,cov,dependency,d,kernel,l)

    def update_cond_mean(self):
        #Y:N*G  mean:N*K
        self.cond_dev = self.Y[np.newaxis,:] - self.mean.transpose()[:,:,np.newaxis]   #K,N,G
        dev = self.cond_dev.copy()
        for i in range(self.kernel.M):
                if len(self.kernel.dependency[i]) > 0:
                    for k in range(self.K):
                        self.cond_dev[k,self.kernel.ss_loc[i],:] = self.cond_dev[k,self.kernel.ss_loc[i],:] - \
                        np.multiply(1/(self.kernel.ds_eig[i][0]+self.delta[k]),self.kernel.A[i]) @ self.kernel.ds_eig[i][1].T @ dev[k,self.kernel.ds_loc[i],:]

    def compute_ll(self,cond_cov_eig):
        ll = np.zeros((self.G,self.K))
        for k in range(self.K):
            ll[:,k] = np.log(2 * np.pi)*self.N + 2*np.log(self.sigma_sq[k])*self.N
            for i in range(self.kernel.M):
                det = np.prod(cond_cov_eig[k][i][0])
                if det <= 0:
                    print(cond_cov_eig[k][i][0]) 
                ll[:,k] += np.log(det)
                temp = self.cond_dev[k][self.kernel.ss_loc[i],:].T @ cond_cov_eig[k][i][1]
                ll[:,k] += np.sum(np.multiply(1/cond_cov_eig[k][i][0],np.square(temp)),axis=1)/self.sigma_sq[k]
        ll = ll*-0.5
        return ll
    
    def update_param(self,omega):
        new_mean= np.zeros_like(self.mean)
        #mean
        for k in range(self.K):
            new_mean[:,k:(k+1)] = self.Y @ omega[:,k:(k+1)] / np.sum(omega[:,k])
        new_dev = self.Y[np.newaxis,:] - new_mean.transpose()[:,:,np.newaxis]
        #pi
        if self.update_pi:
            self.pi = np.average(omega,axis=0)
        
        for k in range(self.K):
            self.cov_new = []
            for i in range(self.kernel.M):
                l = len(self.kernel.ss_loc[i])
                cov_i = np.zeros((l,l))
                for g in range(self.G):
                    cov_i += omega[g,k]*np.outer(new_dev[k,self.kernel.ss_loc[i],g],new_dev[k,self.kernel.ss_loc[i],g])
                cov_i = cov_i/np.sum(omega[:,k])
                self.cov_new.append(cov_i)
            numer,t_2 = 0,0
            for i in range(self.kernel.M):
                numer += np.sum(np.multiply(self.kernel.full_cov[i][i],self.cov_new[i]))
                #denom += np.sum(np.multiply(self.kernel.full_cov[i][i],self.kernel.full_cov[i][i]))
                #t_1 += np.trace(self.kernel.full_cov[i][i])
                t_2 += np.trace(self.cov_new[i])
            #print(numer,denom,t_1,t_2)
            self.sigma_sq[k] = (numer - self.t_1*t_2/self.N)*0.5 / (self.denom - self.t_1**2/self.N) + self.sigma_sq[k]/2
            #print(self.sigma_sq[k],(numer - t_1*t_2/self.N) / (denom - t_1**2/self.N))
            if self.sigma_sq[k] == 0:
                self.delta[k] = t_2*0.5/self.N + self.delta[k]/2
            else:
                self.delta[k] = (t_2-self.sigma_sq[k]*self.t_1)*0.5/(self.N*self.sigma_sq[k]) + self.delta[k]/2
            if self.delta[k]<=0:
                self.delta[k] = 0 
                self.sigma_sq[k] = numer / self.denom 
        return new_mean
    
    def update_mean(self,omega):
        new_mean = self.mean.copy()
        #mean
        for k in range(self.K):
            weight = np.sum(omega[:,k])
            if weight > 0:
                new_mean[:,k] = self.Y @ omega[:,k] / weight
        #pi
        if self.update_pi:
            self.pi = np.average(omega,axis=0)
        return new_mean

    def update_covariance_line_search(self, omega, new_mean, **search_options):
        diagnostics = []
        for k in range(self.K):
            if np.sum(omega[:, k]) <= 0:
                diagnostics.append({'accepted': False, 'reason': 'zero component weight'})
                continue
            result = line_search_covariance(
                self.Y - new_mean[:, k, None], omega[:, k], self.kernel,
                self.sigma_sq[k], self.delta[k], **search_options
            )
            self.sigma_sq[k] = result['sigma_sq']
            self.delta[k] = result['delta']
            diagnostics.append(result)
        return diagnostics
    
    def param_init(self):
        if self.G < self.K:
            raise ValueError("k-means initialization needs at least K features")
        samp_ind = random.sample(range(self.G), max(self.K, self.G//10))
        sample = self.Y[:,samp_ind]
        kmeans = KMeans(n_clusters=self.K, random_state=0).fit(sample.T)
        return kmeans.cluster_centers_.T

    def _run_cluster_frobenius_legacy(self,Y,K,pi=None,mean=None,sigma_sq=None,delta=None,iter=500,threshold=5e-2,init_mean='k_means',update_pi=True):
        #initialization
        self.Y = Y
        self.K = K
        self.N,self.G = self.Y.shape
        self.update_pi = update_pi
        if pi is not None:
            self.pi = np.array(pi,dtype=float)
        else:
            #self.pi = np.random.dirichlet(np.ones(self.K))    
            self.pi = np.ones(self.K,dtype=float) /self.K
        if mean is not None:
            self.mean = np.array(mean,dtype=float)
        else:    
            #self.mean = np.abs(np.random.normal(size=(self.N, self.K)))
            self.mean = np.random.uniform(size=(self.N, self.K))
            if init_mean == 'k_means':
                self.init_mean = self.param_init()
                self.mean = self.init_mean
            elif init_mean == 'sample':
                self.init_mean = self.Y[:,np.random.choice(self.G,self.K)]
            elif isinstance(init_mean,np.ndarray):
                self.init_mean = init_mean
        if sigma_sq is not None:
            self.sigma_sq = np.array(sigma_sq,dtype=float)
        else:
            self.sigma_sq = np.ones(self.K,dtype=float)*0.1
        if delta is not None:
            self.delta = np.array(delta,dtype=float)
        else:
            self.delta = np.ones(self.K,dtype=float)*1
        

        # power = np.zeros((G,self.K))
        self.omega = np.ones((self.G,self.K))/self.K
        converge = False
        count = 0
        #self.ll = np.zeros((self.G,self.K))
        self.denom,self.t_1 = 0,0
        for i in range(self.kernel.M):
            self.denom += np.sum(np.multiply(self.kernel.full_cov[i][i],self.kernel.full_cov[i][i]))
            self.t_1 += np.trace(self.kernel.full_cov[i][i])
        
        while not converge:
            print('Iteration {}'.format(count))
            # cond_cov_eig = (self.kernel,self.delta)
            # cond_dev = update_cond_mean(X,self.mean,self.kernel)

            ll = GaussianNLL(self.Y,self.kernel,self.mean,self.sigma_sq,self.delta)
            #print(self.ll)
            #print(compute_likelihood(self.Y,self.kernel,self.cond_dev[0],self.sigma_sq[0],cond_cov_eig[0]))
            for k in range(self.K):
                if self.pi[k] == 0:
                    self.omega[:,k] = 0
                else:
                    for g in range(self.G):
                        #omega[g,k] = 3/4*self.pi[k]/np.sum(self.pi * _pexp((self.ll[g]-self.ll[g][k])/np.sqrt(self.N))) + omega[g,k]/4
                        self.omega[g,k] = self.pi[k]/np.sum(self.pi * _pexp((ll[g]-ll[g][k])/np.sqrt(self.N))) + 1e-3/self.G
                        # if np.sum(self.pi * _pexp((ll[g]-ll[g][k])/np.sqrt(self.N))) == 0:
                        #     print(ll[g],ll[g][k])

            #print(self.omega)
            new_mean = self.update_mean(self.omega)
            #print(self.delta,self.sigma_sq)
            count += 1
            if count > iter or np.mean(np.abs(new_mean-self.mean))<0.1:
                converge = True
            self.mean = new_mean
            #converge = True
            #print(self.pi,self.sigma_sq,self.delta)
            #indexes = np.array([np.arange(0,2),np.arange(300,302),np.arange(800,802)])
            #print(self.sigma_sq,self.delta,self.pi,omega[indexes])
        #return self.mean
        print('updating variance')
        converge = False
        while not converge:
            print('Iteration {}'.format(count))
            ll = GaussianNLL(self.Y,self.kernel,self.mean,self.sigma_sq,self.delta)

            #print(self.ll)
            #print(compute_likelihood(self.Y,self.kernel,self.cond_dev[0],self.sigma_sq[0],cond_cov_eig[0]))
            for k in range(self.K):
                if self.pi[k] == 0:
                    self.omega[:,k] = 0
                else:
                    for g in range(self.G):
                        #omega[g,k] = 3/4*self.pi[k]/np.sum(self.pi * _pexp((self.ll[g]-self.ll[g][k])/np.sqrt(self.N))) + omega[g,k]/4
                        self.omega[g,k] = self.pi[k]/np.sum(self.pi * _pexp((ll[g]-ll[g][k]))) + 1e-3/self.G
                        # if np.sum(self.pi * _pexp((ll[g]-ll[g][k]))) == 0:
                        #     print(ll[g],ll[g][k])

            #print(self.omega)
            new_mean = self.update_param(self.omega)
            #print(self.delta,self.sigma_sq)
            count += 1
            if count > iter or np.mean(np.abs(new_mean-self.mean))<threshold:
                converge = True
            self.mean = new_mean
            #print(self.pi,self.sigma_sq,self.delta)
        self.labels = np.argmax(self.omega,axis=1)    
        return self.mean

    def run_cluster(self,Y,K,pi=None,mean=None,sigma_sq=None,delta=None,iter=500,
                    threshold=5e-2,init_mean='k_means',update_pi=True,
                    covariance_update='line_search',delta_bounds=(0.0,10.0),
                    scale_floor=1e-8,search_grid_size=12,search_refinements=3,
                    search_xatol=1e-4,acceptance_tolerance=1e-10,
                    likelihood_tolerance=1e-6):
        """Fit the normalized block-conditional mixture.

        The default covariance step profiles sigma_sq and searches delta. The
        Frobenius choice calls the original hybrid fitting routine unchanged.
        """
        if covariance_update == 'frobenius':
            warnings.warn("The legacy Frobenius fit is not a likelihood M-step and "
                          "does not have the line-search safeguard.", RuntimeWarning)
            return self._run_cluster_frobenius_legacy(
                Y,K,pi,mean,sigma_sq,delta,iter,threshold,init_mean,update_pi
            )
        if covariance_update != 'line_search':
            raise ValueError("covariance_update must be 'line_search' or 'frobenius'")
        self.Y = np.asarray(Y, dtype=float)
        if self.Y.ndim != 2 or not np.all(np.isfinite(self.Y)):
            raise ValueError("Y must be a finite locations-by-features matrix")
        self.N,self.G = self.Y.shape
        self.K = int(K)
        if self.N != self.kernel.N or self.G == 0 or not 1 <= self.K <= self.G:
            raise ValueError("K and Y must agree with the kernel and feature count")
        if iter < 1 or threshold < 0 or likelihood_tolerance < 0:
            raise ValueError("invalid iteration or convergence settings")
        self.update_pi = update_pi
        self.pi = (np.asarray(pi,dtype=float) if pi is not None
                   else np.ones(self.K,dtype=float)/self.K)
        if (self.pi.shape != (self.K,) or np.any(self.pi < 0)
                or not np.all(np.isfinite(self.pi)) or not np.isclose(self.pi.sum(),1)):
            raise ValueError("pi must be nonnegative and sum to one")

        if mean is not None:
            self.mean = np.asarray(mean,dtype=float).copy()
        elif isinstance(init_mean,np.ndarray):
            self.mean = np.asarray(init_mean,dtype=float).copy()
        elif init_mean == 'k_means':
            self.mean = self.param_init()
        elif init_mean == 'sample':
            self.mean = self.Y[:,np.random.choice(self.G,self.K,replace=False)].copy()
        else:
            raise ValueError("init_mean must be 'k_means', 'sample', or an array")
        if self.mean.shape != (self.N,self.K) or not np.all(np.isfinite(self.mean)):
            raise ValueError("mean must have shape (locations, K) and be finite")

        self.sigma_sq = (np.asarray(sigma_sq,dtype=float).copy() if sigma_sq is not None
                         else np.full(self.K,0.1))
        self.delta = (np.asarray(delta,dtype=float).copy() if delta is not None
                      else np.ones(self.K))
        if (self.sigma_sq.shape != (self.K,) or not np.all(np.isfinite(self.sigma_sq))
                or np.any(self.sigma_sq < scale_floor)):
            raise ValueError("sigma_sq must be finite and at least scale_floor")
        if (self.delta.shape != (self.K,) or not np.all(np.isfinite(self.delta))
                or np.any(self.delta < delta_bounds[0]) or np.any(self.delta > delta_bounds[1])):
            raise ValueError("initial delta must lie inside delta_bounds")

        search_options = dict(delta_bounds=delta_bounds,scale_floor=scale_floor,
                              grid_size=search_grid_size,refinements=search_refinements,
                              xatol=search_xatol,
                              acceptance_tolerance=acceptance_tolerance)
        self.line_search_diagnostics = []
        self.log_likelihood_history = []
        log_pi = np.where(self.pi > 0,np.log(np.maximum(self.pi,np.finfo(float).tiny)),-np.inf)
        ll = GaussianNLL(self.Y,self.kernel,self.mean,self.sigma_sq,self.delta)
        observed = float(logsumexp(ll + log_pi,axis=1).sum())
        self.log_likelihood_history.append(observed)

        for _ in range(iter):
            log_joint = ll + log_pi
            self.omega = np.exp(log_joint - logsumexp(log_joint,axis=1,keepdims=True))
            previous_mean = self.mean.copy()
            new_mean = self.update_mean(self.omega)
            self.line_search_diagnostics.append(
                self.update_covariance_line_search(self.omega,new_mean,**search_options)
            )
            self.mean = new_mean
            log_pi = np.where(self.pi > 0,np.log(np.maximum(self.pi,np.finfo(float).tiny)),-np.inf)
            ll = GaussianNLL(self.Y,self.kernel,self.mean,self.sigma_sq,self.delta)
            updated = float(logsumexp(ll + log_pi,axis=1).sum())
            self.log_likelihood_history.append(updated)
            relative_change = abs(updated-observed)/max(1.0,abs(observed))
            if (np.mean(np.abs(new_mean-previous_mean)) <= threshold
                    and relative_change <= likelihood_tolerance):
                break
            observed = updated

        log_joint = ll + log_pi
        self.omega = np.exp(log_joint - logsumexp(log_joint,axis=1,keepdims=True))
        self.labels = np.argmax(self.omega,axis=1)
        return self.mean

    def cluster_counts(self,query_label=None):
        if query_label is not None:
            return np.sum(self.labels==query_label)
        else:
            return np.array([np.sum(self.labels==i) for i in range(self.K)])
