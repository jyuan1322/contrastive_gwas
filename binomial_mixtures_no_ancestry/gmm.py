import pickle, math, sys
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm

def gmm(X):
    """
        X is (N x 1)
        gamma is (N x K)
        sigma_sq is (1 x K)
    """

    N = len(X)
    K = 2
    phi = np.ones(K)/K
    mu = np.array([0,0]) # these are not updated
    sigma_sq = np.square(np.array([0.01,0.03]))
    
    keep_iter = True
    LL_old = float("-inf")
    tol = 1e-6
    while keep_iter:
        # E-step
        gamma = np.empty((N,K))
        for n in range(N):
            for k in range(K):
                gamma[n,k] = phi[k] * norm.pdf(X[n], loc=mu[k], scale=np.sqrt(sigma_sq[k]))
        gamma /= gamma.sum(axis=1)[:,None]
        
        # M-step
        phi = np.sum(gamma,axis=0)/N
        sigma_sq = np.dot(np.square(X), gamma)/np.sum(gamma,axis=0)
        
        # calculate likelihood
        LL = 0
        for n in range(N):
            logval = 0.0
            for k in range(K):
                logval += phi[k] * norm.pdf(X[n], loc=mu[k], scale=np.sqrt(sigma_sq[k]))

                # the logval is 0 in some instances
                if math.isnan(logval):
                    sys.exit(0)
            LL += np.log(logval)
        if np.abs(LL - LL_old) < tol:
            keep_iter = False
        LL_old = LL
    return sigma_sq, gamma

if __name__=="__main__":

    X = np.concatenate((np.random.normal(loc=0.0,scale=0.01,size=50),
                       np.random.normal(loc=0.0,scale=0.03,size=50)))
    print(X)
    sigma_sq, gamma = gmm(X)
    print(np.sqrt(sigma_sq))
    print(np.round(gamma))
    num_err0 = np.sum(np.round(gamma)[:50,1])
    num_err1 = np.sum(np.round(gamma)[50:,0])
    print("num err0:", num_err0, "num_err1:", num_err1)
    plt.hist(X,bins=20)
    plt.show()
