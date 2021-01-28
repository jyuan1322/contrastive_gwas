import sys, re, pickle, os, argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from pprint import pprint


def bpflip(snp):
    pair = {"A":"T", "T":"A", "C":"G", "G":"C"}
    return pair[snp]

def getSNPs(SNP_PATH):
    # read SNP file
    # OR and freq pertains to allele 1 (A12)
    snps = {}
    with open(SNP_PATH,"r") as f:
        next(f)
        for line in f:
            # line = line.strip().split(",")
            if not line.startswith("#"):
                line = re.split(''',(?=(?:[^'"]|'[^']*'|"[^"]*")*$)''', line.strip())
                snp_name = line[1]
                if "|" in line[2]: # indel
                    alleles = line[2].split("|")
                else: # snp
                    alleles = [line[2][0], line[2][1]]
                oddsratio = float(line[7].split()[0])
                freq = line[3]
                if oddsratio >= 1:
                    snps[snp_name] = {"eff_allele":alleles[0],
                                      "alt_allele":alleles[1],
                                      "odds_ratio":oddsratio,
                                      "eff_freq":float(freq)}
                else:
                    snps[snp_name] = {"eff_allele":alleles[1],
                                      "alt_allele":alleles[0],
                                      "odds_ratio":1+(1-oddsratio),
                                      "eff_freq":1.0-float(freq)}
    return snps

def extract(SNP_PATH, FILE_PATH, SAMPLE_PATH):
    snps = getSNPs(SNP_PATH)

    # extract phenotypes
    phenos = []
    with open(SAMPLE_PATH,"r") as f:
        next(f)
        next(f)
        for line in f:
            line = line.strip().split()
            pheno = int(line[-1])
            phenos.append(pheno)
    phenos = np.array(phenos)

    # extract genotypes
    genos = {}
    with open(FILE_PATH, "r") as f:
        for line in f: # for each SNP
            line = line.split()
            #print(line[:5])
            snp_name = line[1].split(":")
            if len(snp_name[0]) == 1:
                snp_name = snp_name[0] + ":" + snp_name[1]
            else:
                snp_name = snp_name[0]

            if snp_name in snps:
                # A1 A1, A1 A2, A2 A2
                allele1 = line[3]
                allele2 = line[4]

                data = line[5:]
                num_inds = int(len(data)/3)
                geno = []

                for i in range(num_inds):
                    probs = [float(x) for x in data[i*3:i*3+3]]
                    alcnt = np.argmax(probs) # this counts the number of allele2

                    # account for opposite strand
                    if allele1 == snps[snp_name]["eff_allele"] and allele2 == snps[snp_name]["alt_allele"]:
                        alcnt = 2 - alcnt
                    elif allele1 == snps[snp_name]["alt_allele"] and allele2 == snps[snp_name]["eff_allele"]:
                        pass
                    elif bpflip(allele1) == snps[snp_name]["eff_allele"] and bpflip(allele2) == snps[snp_name]["alt_allele"]:
                        alcnt = 2 - alcnt
                    elif bpflip(allele1) == snps[snp_name]["alt_allele"] and bpflip(allele2) == snps[snp_name]["eff_allele"]:
                        pass
                    else:
                        print(snp_name)
                        print(snps[snp_name])
                        print(allele1, allele2)
                        print(line[:10])
                        print(FILE_PATH)
                        print("-"*20)
                        print(allele1, allele2)
                        print(bpflip(allele1), bpflip(allele2))
                        print(snps[snp_name]["eff_allele"], snps[snp_name]["alt_allele"])
                        sys.exit(0)
                    geno.append(alcnt)

                genos[snp_name] = geno
        geno_array = []
        ors = []
        snp_columns = []
        for i in sorted(genos):
            geno_array.append(genos[i])
            ors.append(snps[i]["odds_ratio"])
            snp_columns.append(i)
        ors = np.array(ors)

        genos = np.array(geno_array)
        genos = genos.T

        ctidxs = np.where(phenos == 1)[0]
        csidxs = np.where(phenos == 2)[0]

        conts = genos[ctidxs,:]
        cases = genos[csidxs,:]
        frqs = np.mean(conts, axis=0) / 2

        # remove columns with variance = 0
        keepcols = []
        snpsdel = 0
        for col in range(cases.shape[1]):
            if np.std(cases[:,col]) != 0 and np.std(conts[:,col]) != 0:
                keepcols.append(col)
            else:
                snpsdel += 1
        conts_abr = conts[:, keepcols]
        cases_abr = cases[:, keepcols]
        ors = ors[keepcols]
        frqs = frqs[keepcols]
        # snp_columns = [snp_columns[i] for i in keepcols]

    # cases_abr has 0-variance SNPs removed
    return cases, conts, cases_abr, conts_abr, snpsdel, ors, frqs, snp_columns

def convertORs(ors, prev):
    betas = []
    prev = 0.01
    thresh = norm.ppf(1-prev, loc=0, scale=1)
    for oddsratio in ors:
        betas.append(norm.ppf(1/(1+np.exp(-(np.log(prev/(1-prev)) + np.log(oddsratio))))) - norm.ppf(prev))
    betas = np.array(betas)
    return betas

def load_data():
    PICKLE_OUT="case_control_matrices.pickle"

    if os.path.exists(PICKLE_OUT):
        super_cases, super_conts, super_betas, snp_columns = pickle.load(open(PICKLE_OUT, "rb"))
    else:

        parser = argparse.ArgumentParser()
        parser.add_argument("--snp-path", dest="snp_path", required=True, help="Summary statistic file")
        parser.add_argument("--geno-path", dest="geno_path", required=True, help="Directory of genotype matrices in Oxford .haps format")
        parser.add_argument("--pheno-path", dest="pheno_path", required=True, help="Directory of .sample files containing phenotype labels")
        parser.add_argument("--file-list", dest="file_list", required=True, help="comma-delimited list of cohort names, .haps files, and .sample files, one line per cohort")
        args = parser.parse_args()

        SNP_PATH=args.snp_path
        GENO_PATH=args.geno_path
        PHENO_PATH=args.pheno_path
        FILE_LIST=args.file_list

        files = []

        with open(FILE_LIST, "r") as f:
            for line in f:
                line = line.strip().split(' ')
                files.append([line[0], GENO_PATH+"/"+line[1], PHENO_PATH+"/"+line[2]])


        prev = 0.01
        thresh = norm.ppf(1-prev, loc=0, scale=1)
        super_cases = None
        super_conts = None
        super_betas = None
        super_frqs = None
        super_hsq = None
        super_snp_cols = None

        for fl in files:
            cht_name, FILE_PATH, SAMPLE_PATH = fl
            # try:
            cases, conts, cases_abr, conts_abr, snpsdel, ors, frqs, snp_columns = extract(SNP_PATH,FILE_PATH,SAMPLE_PATH)
            if super_snp_cols is None:
                super_snp_cols = snp_columns
            else:
                assert super_snp_cols == snp_columns

            # convert odds ratios to liability threshold ratios
            betas = convertORs(ors, prev)
            h_sq = np.sum(np.multiply(np.square(betas), 2*np.multiply(frqs, 1-frqs)))
            if snpsdel == 0: # store for combined set
                super_betas = betas
                super_frqs = frqs
                super_hsq = h_sq

            print("cohort: %s, ncases: %s, nconts: %s, ndel: %s, h_sq: %0.4f" % \
                    (cht_name, cases.shape[0], conts.shape[0], snpsdel, h_sq))

            if super_cases is None:
                super_cases = cases
            else:
                super_cases = np.concatenate((super_cases, cases), axis=0)
            if super_conts is None:
                super_conts = conts
            else:
                super_conts = np.concatenate((super_conts, conts), axis=0)

        pickle.dump((super_cases, super_conts, super_betas, snp_columns),
                    open(PICKLE_OUT, "wb"))

if __name__=="__main__":
    load_data()
