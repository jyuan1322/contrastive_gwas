import pickle, sys, itertools, random, json, copy
# import pystan
import numpy as np
# import statsmodels.api as sm
from numpy.linalg import inv
from matplotlib import pyplot as plt
from scipy.stats import norm, chi2, binom, truncnorm, beta
from scipy.special import digamma,psi,polygamma,gamma,gammaln
from sklearn import mixture
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression, LinearRegression
from pprint import pprint
from product_binomials_em import inference

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

    cases0 = cases0[:num_cases0,:]
    cases1 = cases1[:num_cases1,:]
    print(controls.shape[0], cases0.shape[0], cases1.shape[0])
    return cases0,cases1,controls,betas


def generate_pop_matrix_raw_alleles2(num_snps0, num_snps1, num_cases0, num_cases1, num_controls, gen_hLsq0, gen_hLsq1,
                                    abs_beta=False):
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
    print("rand_coefs:", rand_coefs)

    betas = np.empty((2, V))
    betas[0,:] = np.multiply(betas_temp, rand_coefs)
    betas[1,:] = np.multiply(betas_temp, [1 - x for x in rand_coefs])

    hLsq_calc = np.multiply(np.square(betas),
                            2.0 * np.multiply(ps, (1.0 - ps)))
    hLsq_calc = np.sum(hLsq_calc, axis=1)
    a0 = np.sqrt(hLsq_calc[0] / gen_hLsq0)
    a1 = np.sqrt(hLsq_calc[1] / gen_hLsq1)

    betas[0,:] = betas[0,:] / a0
    betas[1,:] = betas[1,:] / a1

    mean0 = np.dot(betas[0,:], 2*ps)
    mean1 = np.dot(betas[1,:], 2*ps)

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

        # generate sample cases
        samp_conts = np.empty([sample_n, V])
        for i in range(V):
            samp_conts[:,i] = np.random.binomial(2, ps[i], size=sample_n)

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



def simulate(num_snps0=50, num_snps1=50,
             num_cases0=15000, num_cases1=15000, num_controls=30000,
             gen_hLsq0=0.07, gen_hLsq1=0.07, trial=0):

    cases0,cases1,controls,true_betas = generate_pop_matrix_raw_alleles(num_snps0 = num_snps0,
                                                  num_snps1 = num_snps1,
                                                  num_cases0 = num_cases0,
                                                  num_cases1 = num_cases1,
                                                  num_controls = num_controls,
                                                  gen_hLsq0 = gen_hLsq0,
                                                  gen_hLsq1 = gen_hLsq1)

    cases = np.concatenate((cases0,cases1),axis=0)

    # inference here
    print("inference here")
    pi, thetas, LLs = inference(cases)
    with open('models/controls_ncase0_%s_hLsq0_%s_trial_%s.p' % (num_cases0, gen_hLsq0, trial), 'wb') as f:
        pickle.dump(controls,f)
    with open('models/cases_ncase0_%s_hLsq0_%s_trial_%s.p' % (num_cases0, gen_hLsq0, trial), 'wb') as f:
        pickle.dump(cases,f)
    with open('models/true_betas_ncase0_%s_hLsq0_%s_trial_%s.p' % (num_cases0, gen_hLsq0, trial), 'wb') as f:
        pickle.dump(true_betas,f)
    with open('models/pi_ncase0_%s_hLsq0_%s_trial_%s.p' % (num_cases0, gen_hLsq0, trial), 'wb') as f:
        pickle.dump(pi,f)
    with open('models/thetas_ncase0_%s_hLsq0_%s_trial_%s.p' % (num_cases0, gen_hLsq0, trial), 'wb') as f:
        pickle.dump(thetas,f)
    with open('models/LLs_ncase0_%s_hLsq0_%s_trial_%s.p' % (num_cases0, gen_hLsq0, trial), 'wb') as f:
        pickle.dump(LLs,f)

def single_simulation(num_snps0=50, num_snps1=50,
                      num_cases0=15000, num_cases1=15000, num_controls=30000,
                      gen_hLsq0=0.07, gen_hLsq1=0.07, trial=0):


    cases0,cases1,controls,true_betas = generate_pop_matrix_raw_alleles(num_snps0 = num_snps0,
                                                  num_snps1 = num_snps1,
                                                  num_cases0 = num_cases0,
                                                  num_cases1 = num_cases1,
                                                  num_controls = num_controls,
                                                  gen_hLsq0 = gen_hLsq0,
                                                  gen_hLsq1 = gen_hLsq1)

    cases = np.concatenate((cases0,cases1),axis=0)

    """
    print("calculating h2")
    print(calc_h2(cases, controls, prev=0.01))
    print(calc_h2(cases0, controls, prev=0.01))
    sys.exit(0)
    """

    # inference here
    pi, thetas, LLs = inference(cases)
    with open('single_simul.pickle', 'wb') as f:
        pickle.dump((num_controls, num_cases0, num_cases1, controls, cases, true_betas, pi, thetas, LLs), f)
    """
    with open('controls.p', 'wb') as f:
        pickle.dump(controls,f)
    with open('cases.p', 'wb') as f:
        pickle.dump(cases,f)
    with open('true_betas.p', 'wb') as f:
        pickle.dump(true_betas,f)
    with open('pi.p', 'wb') as f:
        pickle.dump(pi,f)
    with open('thetas.p', 'wb') as f:
        pickle.dump(thetas,f)
    with open('LLs.p', 'wb') as f:
        pickle.dump(LLs,f)
    """

def main():

    # NOTE: when splitting snps into mutually exclusive sets, the total h2 is halved in the entire case set
    # confirm this with calc_h2
    h2 = 0.034
    cases_size = 30000
    single_simulation(num_snps0=50, num_snps1=50,
                      num_cases0=int(cases_size/4), num_cases1=int(cases_size/2), num_controls=cases_size,
                      gen_hLsq0=h2, gen_hLsq1=h2, trial=0)
    sys.exit(0)

    # run simulation across a vector of parameters
    # Sample size
    num_trials = 10
    num_cases = [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000] 
    for nc in num_cases:
        for nt in range(num_trials):
            simulate(num_snps0=50, num_snps1=50,
                     num_cases0=nc, num_cases1=nc, num_controls=nc*2,
                     gen_hLsq0=0.07, gen_hLsq1=0.07, trial=nt)
    # Heritability
    num_trials = 10
    herit = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    for ht in herit:
        for nt in range(num_trials):
            simulate(num_snps0=50, num_snps1=50,
                     num_cases0=15000, num_cases1=15000, num_controls=30000,
                     gen_hLsq0=ht, gen_hLsq1=ht, trial=nt)

if __name__=="__main__":
    main()
