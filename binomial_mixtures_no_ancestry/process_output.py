import pickle, sys
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from sklearn.decomposition import PCA
from gmm import gmm
from unsupervised_regression_inference import calc_h2

np.random.seed(0)

def process_results():
    # NOTE: cases and controls may not be the same size
    # num_cases0 = 7500
    # num_cases1 = 15000

    """
    with open('controls.p','rb') as f:
        controls = pickle.load(f)
    with open('cases.p','rb') as f:
        cases = pickle.load(f)
    with open('pi.p','rb') as f:
        pi = pickle.load(f)
    with open('thetas.p','rb') as f:
        thetas = pickle.load(f)
    """
    with open('single_simul.pickle', 'rb') as f:
        num_controls, num_cases0, num_cases1, controls, cases, true_betas, pi, thetas, LLs = pickle.load(f)


    # print("calc h2: %s" % (calc_h2(cases, controls, prev=0.01)))

    af_conts = np.sum(controls,axis=0)/(2.0 * controls.shape[0])
    
    N = cases.shape[0]
    K,M = pi.shape
    cols = ['r','b']
    plt.figure()
    for k in range(K):
        plt.scatter(range(M), pi[k,:] - af_conts, c=cols[k])
    # plt.title('Difference in inferred sub-phenotype AF from control AF')
    plt.axvline(x=50, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('SNPs arranged by sub-type')
    plt.ylabel('Deviation in allele frequency from controls')
    # NOTE: you may need to flip colors and adjust label coordinates
    plt.text(10,-0.045,"true sub-type 0",color='r')
    plt.text(70,-0.045,"true sub-type 1",color='b')

    plt.figure()
    cases0 = cases[:num_cases0,:]
    cases1 = cases[num_cases0:,:]
    # cases0 = cases[:int(N/2),:]
    # cases1 = cases[int(N/2):,:]
    af_cases0 = np.sum(cases0,axis=0)/(2.0 * cases0.shape[0])
    af_cases1 = np.sum(cases1,axis=0)/(2.0 * cases1.shape[0])
    plt.scatter(af_cases0, pi[0,:], c='blue',
                label='sub-pheno 1 vs inferred cluster 1')
    plt.scatter(af_cases1, pi[1,:], c='red',
                label='sub-pheno 2 vs inferred cluster 2')
    plt.title('true vs inferred AFs')
    plt.xlabel('True allele frequencies of known case subsets')
    plt.ylabel('Inferred allele frequency features of inferred clusters')
    plt.legend()

    # probability of membership in each subgroup
    # (cases0, cases1 ordered along x-axis)
    plt.figure()
    cols = ['b','r']
    for k in range(K):
        plt.scatter(range(N),thetas[:,k],s=0.1,alpha=0.5,c=cols[k])
    # plt.title('Inferred sub-phenotype cluster probability.\n' + 
    #           'Blue: sub-pheno 1, Red: sub-pheno 2')
    plt.axvline(x=7500, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Cases arranged by sub-type')
    plt.ylabel('Cluster probability [0,1]')
    plt.text(1000,0.01,"true cases 0",color='r')
    plt.text(12500,0.01,"true cases 1",color='b')

    # get cluster accuracy
    print(thetas.shape)
    acc0 = np.sum(thetas[:num_cases0,0] > thetas[:num_cases0,1])
    acc1 = np.sum(thetas[num_cases0:,1] > thetas[num_cases0:,0])
    print("clust 1: %s / %s" % (acc0, num_cases0))
    print("clust 2: %s / %s" % (acc1, num_cases1))
    print((acc0+acc1)/(num_cases0+num_cases1))
    print("----")
    acc0 = np.sum(thetas[:num_cases0,0] < thetas[:num_cases0,1])
    acc1 = np.sum(thetas[num_cases0:,1] < thetas[num_cases0:,0])
    print("clust 1: %s / %s" % (acc0, num_cases0))
    print("clust 2: %s / %s" % (acc1, num_cases1))
    print((acc0+acc1)/(num_cases0+num_cases1))
    print("NOTE: cluster assignments are interchangeable")

    # PCA of cases
    pca = PCA(n_components=2)
    comps = pca.fit_transform(cases)
    fig, ax = plt.subplots()
    ax.scatter(comps[num_cases0:,0], comps[num_cases0:,1], c='b', s=3, alpha=0.5)
    plt.axvline(x=np.mean(comps[num_cases0:, 0]), color='b', linestyle='--', alpha=0.5)
    ax.scatter(comps[:num_cases0,0], comps[:num_cases0,1], c='r', s=3, alpha=0.5)
    plt.axvline(x=np.mean(comps[:num_cases0,0]), color='r', linestyle='--', alpha=0.5)
    # ax2 = ax.twinx()
    # ax2.hist(comps[num_cases0:,0])
    # ax2.hist(comps[:num_cases0,0])
    print("PCA results")
    print(np.mean(comps[num_cases0:,0]), np.std(comps[num_cases0:,0]))
    print(np.mean(comps[:num_cases0,0]), np.std(comps[:num_cases0,0]))
    print("distance from one mean to the other (not equivalent to the mixture model probs)")
    print(norm.cdf((np.mean(comps[:num_cases0,0]) - np.mean(comps[num_cases0:,0]))/np.std(comps[num_cases0:,0])))
    print(norm.cdf((np.mean(comps[num_cases0:,0]) - np.mean(comps[:num_cases0,0]))/np.std(comps[:num_cases0,0])))
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')

def plot_LLs():
    with open('LLs.p', 'rb') as f:
        LLs = pickle.load(f)
    plt.figure()
    plt.plot(range(len(LLs)), LLs)
    plt.title('EM log likelihood per iteration')
    plt.xlabel('iteration')
    plt.ylabel('LL')

def determine_clusters():
    with open('controls.p','rb') as f:
        controls = pickle.load(f)
    with open('cases.p','rb') as f:
        cases = pickle.load(f)
    with open('pi.p','rb') as f:
        pi = pickle.load(f)
    with open('thetas.p','rb') as f:
        thetas = pickle.load(f)
    
    af_conts = np.sum(controls,axis=0)/(2.0 * controls.shape[0])
    
    N = cases.shape[0]
    K,M = pi.shape
    cols = ['b','r']
    for k in range(K):
        deltas = pi[k,:] - af_conts
        
        sigma_sq, gamma = gmm(deltas)
        print(np.sqrt(sigma_sq))
        # print(np.round(gamma))
        
        num_snps0 = 50
        num_snps1 = 50
        assert (num_snps0+num_snps1) == M
        label_cfgA = np.concatenate((np.tile(np.array([0,1]),(num_snps0,1)),
                                     np.tile(np.array([1,0]),(num_snps1,1))),axis=0)
        label_cfgB = np.concatenate((np.tile(np.array([1,0]),(num_snps0,1)),
                                     np.tile(np.array([0,1]),(num_snps1,1))),axis=0)
        
        label_gam = np.round(gamma)
        snp_grp_acc = np.maximum(np.sum(label_gam[:,0] == label_cfgA[:,0]),
                     np.sum(label_gam[:,0] == label_cfgB[:,0])) / M
        print(snp_grp_acc)
        
        plt.figure()
        grp0_deltas = deltas[:num_snps0]
        grp1_deltas = deltas[num_snps0:]
        bins = np.linspace(-0.05,0.05,20)
        
        if np.mean(np.abs(grp0_deltas)) > np.mean(np.abs(grp1_deltas)):
            grp0_col = 'red'
            grp1_col = 'blue'
        else:
            grp0_col = 'blue'
            grp1_col = 'red'
        plt.hist(grp0_deltas,bins=bins,color=grp0_col,alpha=0.4)
        plt.hist(grp1_deltas,bins=bins,color=grp1_col,alpha=0.4)
        xvals = np.linspace(-0.06,0.06,100)
        null_dist = norm.pdf(xvals, loc=0, scale=np.min(np.sqrt(sigma_sq)))
        effect_dist = norm.pdf(xvals, loc=0, scale=np.max(np.sqrt(sigma_sq)))
        plt.plot(xvals, null_dist, color='blue')
        plt.plot(xvals, effect_dist, color='red')
        plt.title("Nonzero effects classification (Sub-pheno %s)" % k)
        plt.xlabel("SNP allele frequency deviations from controls")
        plt.ylabel("frequency")
        
if __name__=="__main__":
    process_results()
    # plot_LLs()
    # determine_clusters()
    plt.tight_layout()
    plt.show()
