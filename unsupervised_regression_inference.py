import pickle, sys, itertools, random, json, copy, os, pickle
import numpy as np
from random import randint
from numpy.linalg import inv
from matplotlib import pyplot as plt
from scipy.stats import norm, chi2, binom, truncnorm, beta, linregress
from scipy.special import digamma,psi,polygamma,gamma,gammaln
from sklearn import mixture
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.decomposition import PCA
from pprint import pprint
from contrastive_spectral import contrastive_mixture

np.random.seed(0)


def rsquared(x, y):
    x = x.flatten()
    y = y.flatten()
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return r_value**2 * np.sign(slope)

def calc_h2(cases, conts, prev):
    num_cases = cases.shape[0]
    num_conts = conts.shape[0]
    num_indivs = num_cases + num_conts
    P = num_cases / num_indivs
    K = prev
    t = norm.ppf(1-K)

    x = np.concatenate((cases, conts), axis=0)
    y = np.array([1]*cases.shape[0] + [0]*conts.shape[0])

    Gijs = np.corrcoef(x, rowvar=True)
    Zijs = np.outer((y-P), (y-P)) / (P*(1-P))

    Gijs_list = Gijs[np.triu_indices(Gijs.shape[0], k=1)]
    Zijs_list = Zijs[np.triu_indices(Zijs.shape[0], k=1)]

    reg = LinearRegression().fit(Gijs_list.reshape(-1,1), Zijs_list) # fit(X,y)
    slope = reg.coef_[0]
    const = P*(1-P) / (K**2 * (1-K)**2) * norm.pdf(t)**2
    h2 = slope / const

    # print("beta alt: ")
    # print(Zijs_list.dot(Gijs_list) / np.sum(np.square(Gijs_list, Gijs_list)) / const)
    return h2

def generate_pop_matrix_raw_alleles(num_snps0,num_snps1,num_cases0,num_cases1,num_controls,gen_hLsq0,gen_hLsq1,abs_beta=False):
    sample_n = 10000

    V = num_snps0 + num_snps1
    betas_temp = np.empty(V)
    ps = np.random.uniform(size=V)

    # sample betas
    for v in range(V):
        betas_temp[v] = np.random.normal(loc=0.05, scale=0.005)
        if np.random.uniform() < 0.5:
            betas_temp[v] *= -1.0
    
    # rand_coefs = np.random.choice([0.0,1.0], size=V, replace=True)
    # V/2 case0 effects, followed by V/2 case1 effects
    rand_coefs = [0] * num_snps0 + [1] * num_snps1
    print("rand_coefs:",rand_coefs)

    betas = np.empty((2,V))
    betas[0,:] = np.multiply(betas_temp,rand_coefs)
    betas[1,:] = np.multiply(betas_temp,[1 - x for x in rand_coefs])

    hLsq = np.multiply(np.square(betas),
                        2.0*np.multiply(ps,(1.0-ps)))
    hLsq = np.sum(hLsq,axis=1)
    hadd_sq = np.multiply(hLsq, [1.0/gen_hLsq0 - 1, 1.0/gen_hLsq1 - 1])
    print("hLsq:",hLsq)
    print("hadd_sq:",hadd_sq)


    mean0 = np.sum(np.multiply(betas[0,:],2*ps))
    mean1 = np.sum(np.multiply(betas[1,:],2*ps))
    print("means:", mean0, mean1)

    # Y = mu + X^T beta + e
    prev = 0.01
    thresh0 = mean0 + norm.ppf(1.0-prev) * np.sqrt(hadd_sq[0])
    thresh1 = mean1 + norm.ppf(1.0-prev) * np.sqrt(hadd_sq[1])

    cases0 = np.empty([0,V])
    cases1 = np.empty([0,V])
    controls = np.empty([0,V])

    keep_iter = True
    count = 0
    while keep_iter:
        c0N = cases0.shape[0]
        c1N = cases1.shape[0]
        ccN = controls.shape[0]

        count += 1
        print("iter:",count,"controls:",ccN,"cases0:",c0N,"cases1:",c1N)

        # generate sample cases
        samp_conts = np.empty([sample_n,V])
        for i in range(V):
            samp_conts[:,i] = np.random.binomial(2,ps[i],size=sample_n)

        if c0N < num_cases0:
            Y0 = samp_conts.dot(betas[0,:]) + np.random.normal(loc=0.0,
                                                               scale=np.sqrt(hadd_sq[0]),
                                                               size=sample_n)
            Y1 = samp_conts.dot(betas[1,:]) + np.random.normal(loc=0.0,
                                                               scale=np.sqrt(hadd_sq[1]),
                                                               size=sample_n)
            c0 = np.where(Y0 > thresh0)[0]
            c1 = np.where(Y1 > thresh1)[0]
            c0 = set(c0.tolist()) - set(c1.tolist())
            c0 = samp_conts[list(c0),:]
            cases0 = np.concatenate((cases0,c0),axis=0)
            
        if c1N < num_cases1:
            Y0 = samp_conts.dot(betas[0,:]) + np.random.normal(loc=0.0,
                                                               scale=np.sqrt(hadd_sq[0]),
                                                               size=sample_n)
            Y1 = samp_conts.dot(betas[1,:]) + np.random.normal(loc=0.0,
                                                               scale=np.sqrt(hadd_sq[1]),
                                                               size=sample_n)
            c0 = np.where(Y0 > thresh0)[0]
            c1 = np.where(Y1 > thresh1)[0]
            c1 = set(c1.tolist()) - set(c0.tolist())
            c1 = samp_conts[list(c1),:]
            cases1 = np.concatenate((cases1,c1),axis=0)

        if ccN < num_controls:
            Y0 = samp_conts.dot(betas[0,:]) + np.random.normal(loc=0.0,
                                                               scale=np.sqrt(hadd_sq[0]),
                                                               size=sample_n)
            Y1 = samp_conts.dot(betas[1,:]) + np.random.normal(loc=0.0,
                                                               scale=np.sqrt(hadd_sq[1]),
                                                               size=sample_n)
            c0 = np.where(Y0 > thresh0)[0]
            c1 = np.where(Y1 > thresh1)[0]
            cc = set(range(sample_n)) - set(c0.tolist()) - set(c1.tolist())
            cc = samp_conts[list(cc),:]
            controls = np.concatenate((controls,cc[:1000,:]),axis=0)
        
        if c0N >= num_cases0 and c1N >= num_cases1 and ccN >= num_controls:
            keep_iter = False
    
    return cases0,cases1,controls,betas


def generate_pop_matrix_raw_alleles_subpops2(num_snps0, num_snps1, num_cases0, num_cases1, num_controls, gen_hLsq0,
                                            gen_hLsq1, abs_beta=False):
    sample_n = 10000

    V = num_snps0 + num_snps1
    betas_temp = np.empty(V)
    # Prevent occurrence of very rare alleles
    # ps = np.random.uniform(size=V)
    # ps = np.random.uniform(low=0.05, high=0.95, size=V)
    ps = np.random.uniform(low=0.5, high=0.5, size=V)

    # create sub-populations
    fst = 0.1
    # fst = 0.00001 # try checking if you can cluster without ancestry component
    num_subpops = 10
    sub_ps = np.empty((num_subpops, V))
    for v in range(V):
        total_var = ps[v] * (1.0 - ps[v])
        var_s = fst * total_var
        beta_a = -ps[v] * (var_s + ps[v] ** 2 - ps[v]) / var_s
        beta_b = (var_s + ps[v] ** 2 - ps[v]) * (ps[v] - 1) / var_s
        # print("alpha",beta_a,"beta",beta_b)
        for k in range(num_subpops):
            sub_ps[k, v] = np.random.beta(beta_a, beta_b)
            # print("ps:",ps[v],"sub_ps:",sub_ps[k,v])

    # sample betas
    for v in range(V):
        betas_temp[v] = np.random.normal(loc=0.05, scale=0.005)
        # if np.random.uniform() < 0.5:
        #     betas_temp[v] *= -1.0

    # rand_coefs = np.random.choice([0.0,1.0], size=V, replace=True)
    # V/2 case0 effects, followed by V/2 case1 effects
    rand_coefs = [0] * num_snps0 + [1] * num_snps1
    # print("rand_coefs:",rand_coefs)

    betas = np.empty((2, V))
    betas[0, :] = np.multiply(betas_temp, rand_coefs)
    betas[1, :] = np.multiply(betas_temp, [1 - x for x in rand_coefs])

    hLsq_calc = np.multiply(np.square(betas),
                            2.0 * np.multiply(ps, (1.0 - ps)))
    hLsq_calc = np.sum(hLsq_calc, axis=1)
    a0 = np.sqrt(hLsq_calc[0] / gen_hLsq0)
    a1 = np.sqrt(hLsq_calc[1] / gen_hLsq1)

    betas[0, :] = betas[0, :] / a0
    betas[1, :] = betas[1, :] / a1

    mean0 = np.dot(betas[0, :], 2 * ps)
    mean1 = np.dot(betas[1, :], 2 * ps)

    # Y = mu + X^T beta + e
    prev = 0.01
    thresh = norm.ppf(1.0 - prev)

    cases0 = np.empty([0, V])
    cases1 = np.empty([0, V])
    controls = np.empty([0, V])

    keep_iter = True
    count = 0
    while keep_iter:
        c0N = cases0.shape[0]
        c1N = cases1.shape[0]

        ccN = controls.shape[0]

        count += 1
        print("iter:", count, "controls:", ccN, "cases0:", c0N, "cases1:", c1N)

        # generate sample cases, sampling a particular sub-population randomly
        samp_conts = np.empty([sample_n, V])
        sub_grps = np.random.randint(0, num_subpops, size=sample_n)
        for i in range(V):
            # samp_conts[:,i] = np.random.binomial(2,ps[i],size=sample_n)
            pss = [sub_ps[k, i] for k in sub_grps]
            samp_conts[:, i] = np.random.binomial(2, pss, size=sample_n)

        if c0N < num_cases0:
            Y0 = samp_conts.dot(betas[0, :]) - mean0 + np.random.normal(loc=0.0,
                                                                        scale=np.sqrt(1 - gen_hLsq0),
                                                                        size=sample_n)
            Y1 = samp_conts.dot(betas[1, :]) - mean1 + np.random.normal(loc=0.0,
                                                                        scale=np.sqrt(1 - gen_hLsq1),
                                                                        size=sample_n)
            c0 = np.where(Y0 > thresh)[0]
            c1 = np.where(Y1 > thresh)[0]

            c0 = set(c0.tolist()) - set(c1.tolist())
            c0 = samp_conts[list(c0), :]
            cases0 = np.concatenate((cases0, c0), axis=0)

        if c1N < num_cases1:
            Y0 = samp_conts.dot(betas[0, :]) - mean0 + np.random.normal(loc=0.0,
                                                                        scale=np.sqrt(1 - gen_hLsq0),
                                                                        size=sample_n)
            Y1 = samp_conts.dot(betas[1, :]) - mean1 + np.random.normal(loc=0.0,
                                                                        scale=np.sqrt(1 - gen_hLsq1),
                                                                        size=sample_n)
            c0 = np.where(Y0 > thresh)[0]
            c1 = np.where(Y1 > thresh)[0]
            c1 = set(c1.tolist()) - set(c0.tolist())
            c1 = samp_conts[list(c1), :]

            cases1 = np.concatenate((cases1, c1), axis=0)

        if ccN < num_controls:
            Y0 = samp_conts.dot(betas[0, :]) - mean0 + np.random.normal(loc=0.0,
                                                                        scale=np.sqrt(1 - gen_hLsq0),
                                                                        size=sample_n)
            Y1 = samp_conts.dot(betas[1, :]) - mean1 + np.random.normal(loc=0.0,
                                                                        scale=np.sqrt(1 - gen_hLsq1),
                                                                        size=sample_n)
            c0 = np.where(Y0 > thresh)[0]
            c1 = np.where(Y1 > thresh)[0]
            cc = set(range(sample_n)) - set(c0.tolist()) - set(c1.tolist())
            cc = samp_conts[list(cc), :]
            controls = np.concatenate((controls, cc[:1000, :]), axis=0)

        if c0N >= num_cases0 and c1N >= num_cases1 and ccN >= num_controls:
            keep_iter = False

    return cases0, cases1, controls, betas


def single_simulation(num_snps0=50, num_snps1=50,
                      num_cases0=15000, num_cases1=15000, num_controls=30000,
                      gen_hLsq0=0.07, gen_hLsq1=0.07, trial=0):

    PICKLE_FILE = "case_control_simul.p"
    if os.path.exists(PICKLE_FILE):
        results = pickle.load(open(PICKLE_FILE, "rb"))
        cases0 = results["cases0"]
        cases1 = results["cases1"]
        controls = results["controls"]
        true_betas = results["true_betas"]
    else:
        cases0,cases1,controls,true_betas = generate_pop_matrix_raw_alleles_subpops2(
                                                  num_snps0 = num_snps0,
                                                  num_snps1 = num_snps1,
                                                  num_cases0 = num_cases0,
                                                  num_cases1 = num_cases1,
                                                  num_controls = num_controls,
                                                  gen_hLsq0 = gen_hLsq0,
                                                  gen_hLsq1 = gen_hLsq1)
        with open(PICKLE_FILE, "wb") as f:
            pickle.dump({"cases0":cases0,
                         "cases1":cases1,
                         "controls":controls,
                         "true_betas":true_betas}, f)


    cases = np.concatenate((cases0,cases1),axis=0)

    # identify ancestry by PCA
    pca = PCA(n_components=1)
    pca.fit(controls)
    # pca_comp = np.squeeze(pca.components_) # single comp
    pca_comp = pca.components_
    # print(pca_comp)
    # print("pca shape:", pca_comp.shape)

    cont_afs = np.sum(controls,axis=0)/(2*controls.shape[0])
    cont_afs = np.expand_dims(np.array(cont_afs),axis=0)


    a,lam = contrastive_mixture(cases,controls,K=2,gamma=1.0,num_iter=100)
    #for av in a:
    #    av = av / np.linalg.norm(av, ord=1)
    print("contrastive mixture output:")
    for i in range(len(a)):
        print("*"*50)
        print("component of a")
        print(a[i])
        print(lam[i])

    # get the true components
    truepi_0 = np.sum(cases0, axis=0)/(2*cases0.shape[0])
    truepi_1 = np.sum(cases1, axis=0)/(2*cases1.shape[0])
    true_binom_comp = np.squeeze(truepi_1 - truepi_0)


    print("dot product:")
    print("w/ pheno:", np.dot(true_binom_comp,a[0]))
    print("w/ pca:", np.dot(pca_comp,a[0]))
    
    plt.figure()
    plt.scatter(true_binom_comp, a[0], label="comp 1", c='b')
    plt.scatter(true_binom_comp, a[1], label="comp 2", c='r')
    # plt.scatter(truepi_0 - np.mean(cases, axis=0)/2, a[0] - (a[0] + a[1])/2, label="comp 1")
    # plt.scatter(truepi_1 - np.mean(cases, axis=0)/2, a[1] - (a[0] + a[1])/2, label="comp 2")
    # plt.xlabel("true cluster diff")
    plt.xlabel(r"true subtype allele freq $\Delta$")
    # plt.ylabel("learned components")
    plt.ylabel("inferred contrastive vectors")
    # plt.legend()
    
    plt.figure()
    plt.scatter(pca_comp, a[0], label="comp 1", c='b')
    plt.scatter(pca_comp, a[1], label="comp 2", c='r')
    plt.xlabel("first principal component (control)")
    # plt.ylabel("learned components")
    plt.ylabel("inferred contrastive vectors")
    # plt.legend()

    plt.figure()
    plt.scatter(pca_comp, true_binom_comp)
    plt.xlabel("PCA (control) components")
    plt.ylabel("true cluster diff")

    contrastive_r2 = max(rsquared(true_binom_comp, a[0]), rsquared(true_binom_comp, a[1]))
    pca_r2 = max(rsquared(pca_comp, a[0]), rsquared(pca_comp, a[1]))
    print("r squareds")
    print(contrastive_r2)
    print(pca_r2)

    plt.show()


def test_h2():
    num_snps0 = 10
    num_snps1 = 10
    num_cases0 = 1500
    num_cases1 = 1500
    num_controls = 3000
    hl2s = [0.1]
    num_trials = 50
    f = open("performance_hl2.txt", "a", buffering=1)
    for hl2 in hl2s:
        for nt in range(num_trials):
            cases0,cases1,controls,true_betas = generate_pop_matrix_raw_alleles_subpops2(
                                                      num_snps0 = num_snps0,
                                                      num_snps1 = num_snps1,
                                                      num_cases0 = num_cases0,
                                                      num_cases1 = num_cases1,
                                                      num_controls = num_controls,
                                                      gen_hLsq0 = hl2,
                                                      gen_hLsq1 = hl2)

            cases = np.concatenate((cases0,cases1),axis=0)

            h2_res = calc_h2(cases, controls, prev=0.01)
            f.write(str(hl2) + "," + str(nt) + "," + str(h2_res) + "\n")
    f.close()

# an hL^2 of 0.1 for 10 SNPs achieves .00358 variance explained, which is the correct per-SNP varexp
def plot_h2():
    with open("performance_hl2.txt", "r") as f:
        vals = {}
        for line in f:
            line = [float(x) for x in line.split(",")]
            h2 = line[0]
            nt = line[1]
            h2_calc = line[2]
            if h2 not in vals:
                vals[h2] = []
            vals[h2].append(h2_calc)
        means = []
        stds = []
        print(vals)
        for h2 in sorted(vals.keys()):
            means.append(np.mean(vals[h2]))
            stds.append(np.std(vals[h2]))
        print(means)
        print(stds)
        plt.errorbar(sorted(vals.keys()), means, yerr=stds)
        plt.show()
def main():
    # test_h2()
    # plot_h2()
    # sys.exit(0)


    single_simulation(num_snps0=10, num_snps1=10,
                      num_cases0=1500, num_cases1=1500, num_controls=3000,
                      gen_hLsq0=0.1, gen_hLsq1=0.1, trial=0)

if __name__=="__main__":
    main()
