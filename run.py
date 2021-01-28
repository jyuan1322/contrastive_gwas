import sys, os, pickle
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from contrastive_spectral import contrastive_mixture

def eval_prob(geno, a):
    llhd = np.dot(geno, np.log(a)) + np.dot((2-geno), np.log(1-a)) + \
            np.log(2) * np.sum((geno==1))
    return llhd

if __name__=="__main__":
    np.random.seed(0)

    PICKLE_OUT="case_control_matrices.pickle"
    cases, conts, betas, snp_columns = pickle.load(open(PICKLE_OUT, "rb"))

    pca = PCA(n_components=1)
    pca.fit(conts)
    pca_comp = pca.components_

    pca2 = PCA(n_components=2)
    pca2.fit(conts)
    pca_comp2 = pca2.components_

    num_snps = cases.shape[1]
    num_case_comps = 3
    colors = ['b', 'r', 'g']
    # for gamma in np.linspace(0, 3, 20):
    for gamma in [0.1]:
        a, lam = contrastive_mixture(cases, conts, K=num_case_comps, gamma=gamma, num_iter=100)

        print(np.mean(cases, axis=0) - np.mean(conts, axis=0))
        fig, ax = plt.subplots()
        for comp in range(num_case_comps):
            ax.scatter(range(num_snps), a[comp] - np.mean(cases, axis=0), label="comp %s" % (comp))
        ax.set_xlabel("SNP index")
        ax.set_ylabel("component weight")
        fig.legend()

        fig, ax = plt.subplots()
        for k in range(num_case_comps):
            ax.scatter(pca_comp, a[k], label="comp %s" % (k), c=colors[k])
        ax.set_xlabel("first PC")
        ax.set_ylabel("cluster allele frequency")
        ax.legend()

        # check component weights against MAF
        cases_idx = []
        for k in range(num_case_comps):
            cases_idx.append([])
        for i in range(cases.shape[0]):
            probs = [eval_prob(cases[i, :], a[k]) for k in range(num_case_comps)]
            max_idx = probs.index(max(probs))
            cases_idx[max_idx].append(i)
        print([len(x) for x in cases_idx])
        fig, ax = plt.subplots()
        for k in range(num_case_comps):
            ax.scatter(np.mean(conts, axis=0)/2, a[k], c=colors[k], label="comp %s (%s)" % (k, len(cases_idx[k])))
        ax.legend()
        lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
                np.max([ax.get_xlim(), ax.get_ylim()])]
        ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        ax.set_xlabel("mean control allele frequency")
        ax.set_ylabel("cluster allele frequency")
        # ax.scatter(np.mean(conts, axis=0), np.mean(cases, axis=0))

        # check component weights against diff between cases and controls
        fig, ax = plt.subplots()
        for k in range(num_case_comps):
            ax.scatter((np.mean(cases, axis=0) - np.mean(conts, axis=0)) / 2, a[k])
        ax.set_xlabel("mean diff (case-cont) allele frequency")
        ax.set_ylabel("inferred cluster allele freq")

        # assign labels to controls
        conts_idx = []
        for k in range(num_case_comps):
            conts_idx.append([])
        for i in range(conts.shape[0]):
            probs = [eval_prob(conts[i,:], a[k]) for k in range(num_case_comps)]
            max_idx = probs.index(max(probs))
            conts_idx[max_idx].append(i)
        print([len(x) for x in conts_idx])

        trans = pca2.transform(conts)
        fig, ax = plt.subplots()
        for i,k in enumerate(conts_idx):
            sub_group = trans[k, :]
            ax.scatter(sub_group[:,0], sub_group[:,1], alpha=0.5, s=10, c=colors[i], edgecolors='none')
        ax.set_xlabel("Controls PC1")
        ax.set_ylabel("Controls PC2")
        plt.show()

    sys.exit(0)

    """
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
    """
