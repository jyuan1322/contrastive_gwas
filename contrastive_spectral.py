import sys
import numpy as np

"""
def m2(c):
    N,l = c.shape
    sumN = np.zeros((l,l))
    for i in range(N):
        sumN += np.outer(c[i,:],c[i,:]) - np.diag(c[i,:])
    # return 1.0/(N * l * (l-1)) * sumN / 4
    return 1.0/(N * l * (l-1)) * sumN
"""

def m2(c):
    N,l = c.shape
    sumN = np.zeros((l,l))
    # c_cent = c - np.mean(c, axis=0)
    for i in range(N):
        sumN += np.outer(c[i,:],c[i,:])
    # return 1.0/N * sumN / 4
    return 1.0/N * sumN / 2

"""
def m3(v): # v is a 1d vector; for cf or cb, input the 1xD SNP means
    d = v.shape[0]
    a = np.outer(v,v)
    a = a.reshape(d,d,1)
    b = v.reshape(1,d)
    # return 1.0/(d * (d-1) * (d-2)) * np.dot(a,b) / 8
    print(d)

    return 1.0/(d * (d-1) * (d-2)) * np.dot(a,b)
"""

def m3(c):
    N,l = c.shape
    sumN = np.zeros((l,l,l))
    # c_cent = c - np.mean(c, axis=0)
    for i in range(N):
        if i % 100 == 0:
            print("%s / %s" % (i,N))
        v = c[i,:]
        a = np.outer(v, v)
        a = a.reshape(l,l,1)
        b = v.reshape(1,l)
        sumN += np.dot(a,b)
    # return 1.0/N * sumN / 8
    return 1.0/N * sumN / 4

"""
def m3(c,v):
    # estimate M^f_3(I,v,v)
    if len(c.shape > 1):
        N,l = c.shape
    else:
        N = c.shape[0]
        l = 1
    sumN = np.zeros(l)
    for i in range(N):
        sumN += np.dot(c[i,:],v)**2 * c[i,:] - \
                2 * np.dot(c[i,:],v) * np.multiply(c[i,:],v) - \
                np.dot(c[i,:], np.multiply(v,v)) * c[i,:] + \
                2 * np.multiply(c[i,:], np.multiply(v,v))
    return 1.0/(N * l * (l-1) * (l-2)) * sumN


def deflation(lam,a,v):
    d = 0
    for t in range(len(lam)):
        d += lam[t] * np.dot(a[t],v)**2 * a[t]
    return d
"""



def tensor_power(cf, cb, gamma, K, num_iter):

    print("step 1")

    M2 = m2(cf) - gamma * m2(cb)
    D = M2.shape[0]
    u,s,vh = np.linalg.svd(M2, full_matrices=True)

    print("step 2")

    M2p = np.linalg.multi_dot([vh[:K,:].T,
                        np.diag(1.0/s[:K]),
                        u[:,:K].T])

    print("step 3")

    # cp = np.mean(cf,axis=0) - gamma * np.mean(cb,axis=0)
    # T = m3(cp)
    T = m3(cf) - gamma * m3(cb)

    print("step 4")

    a = []
    lam = []
    for t in range(K):
        uu = np.dot(M2, np.random.uniform(low=-1,high=1,size=D))
        # uu = np.random.uniform(low=np.min(M2),
        #                        high=np.max(M2),
        #                        size=D)
        print(uu.shape)

        print("calculating M3")
        for i in range(num_iter):
            Muu = np.dot(M2p,uu)
            print("iter:%s" % (i))
            print("-"*50)
            #dfl = deflation(lam,a,Muu)
            # uu = m3(cf,Muu) - gamma * m3(cb,Muu) - dfl
            # power iteration
            uu = np.tensordot(T,np.outer(Muu,Muu),axes=([1,2],[0,1]))
            uu /= np.linalg.norm(uu)

        a.append(uu/np.sqrt(np.abs(np.dot(uu, np.dot(M2p,uu)))))
        M2pa = np.dot(M2p,a[t])
        #dfl = deflation(lam,a[:t],M2pa) # penalty up to t-1
        #lam.append(m3(M2pa,M2pa) - gamma * m3(M2pa,M2pa) - dfl)
        
        
        # lam.append(np.tensordot(T,m3(M2pa),axes=([0,1,2],[0,1,2])))
        # check lambda *** Original calculation is wrong
        d = len(uu)
        lsum = 0
        for i in range(d):
            for j in range(d):
                for k in range(d):
                    lsum += T[i,j,k]*M2pa[i]*M2pa[j]*M2pa[k]
        # lam[-1] = lsum
        lam.append(lsum)

        # T = T - lam[t] * m3(a[t])
        d = a[t].shape[0]
        cr1 = np.outer(a[t],a[t])
        cr1 = cr1.reshape(d,d,1)
        cr2 = a[t].reshape(1,d)
        T = T - lam[t] * np.dot(cr1,cr2)

    return a,lam


def contrastive_mixture(cf, cb, K, gamma=0.1, num_iter=10):

    # gamma estimated from prior belief about shared clusters
    a,lam = tensor_power(cf, cb, gamma, K, num_iter)
    return a,lam

if __name__=="__main__":
    cases0 = np.random.multivariate_normal([0,1], np.eye(2), size=1000)
    cases1 = np.random.multivariate_normal([1,0], np.eye(2), size=1000)
    cases = np.concatenate((cases0,cases1), axis=0)
    conts = np.random.multivariate_normal([0,0], np.eye(2), size=1000)

    print(cases.shape)
    print(conts.shape)

    a, lam = contrastive_mixture(cases, conts, K=2, gamma=0.2, num_iter=20)
    for i in range(len(a)):
        a[i] /= np.linalg.norm(a[i], ord=1)
        lam[i] = 1.0/(lam[i]**2)
    print("-"*50 + "\n")
    print(a)
    print(lam)
