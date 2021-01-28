import sys, os, pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from scipy.stats import norm, linregress
from sklearn.decomposition import PCA
from contrastive_spectral import contrastive_mixture

def rsquared(x, y):
    x = x.flatten()
    y = y.flatten()
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return r_value**2 * np.sign(slope)

def generate_pop_matrix_raw_alleles_subpops(num_snps0, num_snps1, num_cases0, num_cases1, num_controls, gen_hLsq0,
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

    hLsq = np.multiply(np.square(betas),
                       2.0 * np.multiply(ps, (1.0 - ps)))
    hLsq = np.sum(hLsq, axis=1)
    hadd_sq = np.multiply(hLsq, [1.0 / gen_hLsq0 - 1, 1.0 / gen_hLsq1 - 1])
    # print("hLsq:",hLsq)
    # print("hadd_sq:",hadd_sq)

    mean0 = np.sum(np.multiply(betas[0, :], 2 * ps))
    mean1 = np.sum(np.multiply(betas[1, :], 2 * ps))
    # print("means:", mean0, mean1)

    # Y = mu + X^T beta + e
    prev = 0.01
    thresh0 = mean0 + norm.ppf(1.0 - prev) * np.sqrt(hadd_sq[0])
    thresh1 = mean1 + norm.ppf(1.0 - prev) * np.sqrt(hadd_sq[1])

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
            Y0 = samp_conts.dot(betas[0, :]) + np.random.normal(loc=0.0,
                                                                scale=np.sqrt(hadd_sq[0]),
                                                                size=sample_n)
            Y1 = samp_conts.dot(betas[1, :]) + np.random.normal(loc=0.0,
                                                                scale=np.sqrt(hadd_sq[1]),
                                                                size=sample_n)
            c0 = np.where(Y0 > thresh0)[0]
            c1 = np.where(Y1 > thresh1)[0]

            c0 = set(c0.tolist()) - set(c1.tolist())
            c0 = samp_conts[list(c0), :]
            cases0 = np.concatenate((cases0, c0), axis=0)

        if c1N < num_cases1:
            Y0 = samp_conts.dot(betas[0, :]) + np.random.normal(loc=0.0,
                                                                scale=np.sqrt(hadd_sq[0]),
                                                                size=sample_n)
            Y1 = samp_conts.dot(betas[1, :]) + np.random.normal(loc=0.0,
                                                                scale=np.sqrt(hadd_sq[1]),
                                                                size=sample_n)
            c0 = np.where(Y0 > thresh0)[0]
            c1 = np.where(Y1 > thresh1)[0]
            c1 = set(c1.tolist()) - set(c0.tolist())
            c1 = samp_conts[list(c1), :]

            cases1 = np.concatenate((cases1, c1), axis=0)

        if ccN < num_controls:
            Y0 = samp_conts.dot(betas[0, :]) + np.random.normal(loc=0.0,
                                                                scale=np.sqrt(hadd_sq[0]),
                                                                size=sample_n)
            Y1 = samp_conts.dot(betas[1, :]) + np.random.normal(loc=0.0,
                                                                scale=np.sqrt(hadd_sq[1]),
                                                                size=sample_n)
            c0 = np.where(Y0 > thresh0)[0]
            c1 = np.where(Y1 > thresh1)[0]
            cc = set(range(sample_n)) - set(c0.tolist()) - set(c1.tolist())
            cc = samp_conts[list(cc), :]
            controls = np.concatenate((controls, cc[:1000, :]), axis=0)

        if c0N >= num_cases0 and c1N >= num_cases1 and ccN >= num_controls:
            keep_iter = False

    return cases0, cases1, controls, betas

def single_simulation(num_snps0=50, num_snps1=50,
                      num_cases0=15000, num_cases1=15000, num_controls=30000,
                      gen_hLsq0=0.07, gen_hLsq1=0.07):


    cases0, cases1, controls, true_betas = generate_pop_matrix_raw_alleles_subpops(
                                                                        num_snps0=num_snps0,
                                                                        num_snps1=num_snps1,
                                                                        num_cases0=num_cases0,
                                                                        num_cases1=num_cases1,
                                                                        num_controls=num_controls,
                                                                        gen_hLsq0=gen_hLsq0,
                                                                        gen_hLsq1=gen_hLsq1)
    cases0 = cases0[:num_cases0,:]
    cases1 = cases1[:num_cases1,:]
    return cases0, cases1, controls

def contrastive_r2_test(cases0, cases1, controls, gamma):
    cases = np.concatenate((cases0, cases1), axis=0)

    # identify ancestry by PCA
    pca = PCA(n_components=1)
    pca.fit(controls)
    # pca_comp = np.squeeze(pca.components_) # single comp
    pca_comp = pca.components_

    # inference here
    print("inference here")

    a, lam = contrastive_mixture(cases, controls, K=2, gamma=gamma, num_iter=100)
    # for av in a:
    #    av = av / np.linalg.norm(av, ord=1)
    print("contrastive mixture output:")
    for i in range(len(a)):
        print("*" * 50)
        print("component of a")
        print(a[i])
        print(lam[i])

    # get the true components
    truepi_0 = np.sum(cases0, axis=0) / (2 * cases0.shape[0])
    truepi_1 = np.sum(cases1, axis=0) / (2 * cases1.shape[0])
    true_binom_comp = np.squeeze(truepi_1 - truepi_0)

    contrastive_r2 = max(rsquared(true_binom_comp, a[0]), rsquared(true_binom_comp, a[1]))
    pca_r2 = max(rsquared(pca_comp, a[0]), rsquared(pca_comp, a[1]))
    return contrastive_r2, pca_r2

def performance():
    gammas = np.arange(start=0.1, stop=1.5, step=0.1)
    case_sizes = [1000, 1500, 2000, 2500, 3000, 3500, 4000]
    # h2s = [0.1, 0.05, 0.025, 0.0125, .00625]
    h2 = 0.1
    num_subtype_snps = 10
    num_trials = 20

    f = open("performance_data.txt", "a", buffering=1)
    for case_size in case_sizes:
        for nt in range(num_trials):
            cases0, cases1, controls = single_simulation(num_snps0=num_subtype_snps,
                                                         num_snps1=num_subtype_snps,
                                                         num_cases0=int(case_size/2),
                                                         num_cases1=int(case_size/2),
                                                         num_controls=case_size,
                                                         gen_hLsq0=h2, gen_hLsq1=h2)
            for gamma in gammas:
                contrastive_r2, pca_r2 = contrastive_r2_test(cases0, cases1, controls, gamma)
                result_string = ",".join([str(x) for x in [case_size, nt, gamma, contrastive_r2, pca_r2]])
                print(result_string)
                f.write(result_string + "\n")
    f.close()

if __name__=="__main__":
    # performance()
    with open("performance_data.txt", "r") as f:
        results = {}
        for line in f:
            line = [float(x) for x in line.split(",")]
            case_size = line[0]
            nt = line[1]
            gamma = line[2]
            contrastive_r2 = line[3]
            pca_r2 = line[4]

            if case_size not in results:
                results[case_size] = {}
            if gamma not in results[case_size]:
                results[case_size][gamma] = []
            results[case_size][gamma].append((contrastive_r2, pca_r2))
        case_sizes = sorted(results.keys())
        gammas = sorted(results[case_sizes[0]].keys())

        mingamma = 0.4
        maxgamma = 1.1
        cm = plt.get_cmap('RdYlBu')
        cNorm = colors.Normalize(vmin=mingamma, vmax=maxgamma)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

        fig, ax = plt.subplots()
        for gamma in gammas:
            if gamma >= mingamma and gamma <= maxgamma:
                colorVal = scalarMap.to_rgba(gamma)
                diff_means = []
                diff_stds = []

                for case_size in case_sizes:
                    diff_means.append(np.mean([x[0]-x[1] for x in results[case_size][gamma]]))
                    diff_stds.append(np.std([x[0]-x[1] for x in results[case_size][gamma]]))

                plt.errorbar(case_sizes, diff_means, yerr=diff_stds,
                             c=colorVal, capsize=5, label=np.round(gamma, decimals=2))
        # plt.colorbar(scalarMap)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
                  title=r"$\gamma$")

        # plt.legend()
        plt.xlabel("case/control sample size")
        plt.ylabel(r"$\Delta r^2$ (contrastive - PCA)")

        plt.figure()
        case_size = case_size
        ctrstv_means = []
        ctrstv_stds = []
        pca_means = []
        pca_stds = []
        diff_means = []
        diff_stds = []
        for gamma in gammas:
            ctrstv_means.append(np.mean([x[0] for x in results[case_size][gamma]]))
            ctrstv_stds.append(np.std([x[0] for x in results[case_size][gamma]]))
            pca_means.append(np.mean([x[1] for x in results[case_size][gamma]]))
            pca_stds.append(np.std([x[1] for x in results[case_size][gamma]]))
            diff_means.append(np.mean([x[0] - x[1] for x in results[case_size][gamma]]))
            diff_stds.append(np.std([x[0] - x[1] for x in results[case_size][gamma]]))
        # plt.errorbar(gammas, diff_means, yerr=diff_stds, capsize=5, c='k')
        plt.errorbar(gammas, ctrstv_means, yerr=ctrstv_stds, capsize=5, c='g')
        plt.errorbar(gammas, pca_means, yerr=pca_stds, capsize=5, c='r')
        plt.xlabel(r"$\gamma$")
        plt.ylabel(r"$r^2$")
        plt.show()