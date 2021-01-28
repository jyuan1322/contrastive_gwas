import sys, timeit
import numpy as np
from scipy.special import gamma
from matplotlib import pyplot as plt

import warnings
warnings.simplefilter('error')

def inference(cases, timing=False):
    
    N,M = cases.shape
    K = 2 # number of clusters
    
    # initialize h(j|xh) (N x K)
    h_post = np.random.uniform(size=(N,K))
    
    # approximate nu_j (K x 1), pi_ij (K x M)
    keep_iter = True
    count = 0
    L_old = float("-inf")
    LLs = []
    tol = 5e-2

    while(keep_iter):
        hj_tot = np.sum(h_post,axis=0)        
        nu = hj_tot / N
        if timing:
            t1 = timeit.default_timer()
        pi = np.dot(h_post.T, cases)
        if timing:
            print("-"*50)
            print("T1", timeit.default_timer() - t1)

        for k in range(K):
            pi[k,:] /= (2.0*hj_tot[k])

        if timing:
            t2 = timeit.default_timer()
        # update posteriors
        h_new = np.empty((h_post.shape))
        for n in range(N):
            for k in range(K):
                snp_vec = np.multiply(np.power(pi[k,:], cases[n,:]),
                                      np.power(1-pi[k,:], 2-cases[n,:]))
                comb = np.ones(M)
                comb[np.where(cases[n,:] == 1)[0]] = 2
                snp_vec = np.multiply(snp_vec, comb)
                
                h_new[n,k] = nu[k] * np.prod(snp_vec)

        h_new = h_new/h_new.sum(axis=1)[:,None]
        h_post = h_new
        if timing:
            print("T2", timeit.default_timer() - t2)

        if timing:
            t3 = timeit.default_timer()
        count += 1
        if count % 1 == 0:
            # calculate log likelihood
            L = 0.0
            for n in range(N):
                logsum = 0.0
                for k in range(K):
                    snp_vec = np.multiply(np.power(pi[k,:], cases[n,:]),
                                          np.power(1-pi[k,:], 2-cases[n,:]))
                    comb = np.ones(M)
                    comb[np.where(cases[n,:] == 1)[0]] = 2
                    snp_vec = np.multiply(snp_vec, comb)
                    logsum += nu[k] * np.prod(snp_vec)

                L += np.log(logsum)

            print("iter:",count, "LL:",L)
            if np.abs(L - L_old) < tol:
                keep_iter = False
            LLs.append(L)
            L_old = L
        if timing:
            print("T3", timeit.default_timer() - t3)
    return pi, h_post, LLs
