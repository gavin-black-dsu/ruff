import numpy as np
from scipy.stats import ks_2samp
from tqdm import tqdm, trange

def estimate_power_with_data(data1, data2, n_simulations=1000, alpha=0.05):
    n1 = len(data1)
    n2 = len(data2)
    #n1, n2 = 100, 100
    rejections = 0
    
    for _ in range(n_simulations):
        # Randomly sample from data1 and data2 with replacement
        sample1 = np.random.choice(data1, size=n1, replace=True)
        sample2 = np.random.choice(data2, size=n2, replace=True)
        
        # Perform the KS test
        #print(sample1, sample2)
        statistic, p_value = ks_2samp(sample1, sample2)
        if p_value < alpha:
            rejections += 1
    
    return rejections / n_simulations


def perform_ks(data1, data2, alpha, verbose=False):
    # Perform the initial Kolmogorov-Smirnov test
    initial_statistic, initial_p_value = ks_2samp(data1, data2)

    # Display initial test results
    if verbose:
        print(f"Initial KS statistic: {initial_statistic}")
        print(f"Initial P-value: {initial_p_value}")


    hypothesis = 0
    # Decision based on initial p-value
    if initial_p_value < alpha:
        hypothesis = 1
        if verbose: print("H1: Reject null hypothesis - significant difference between distributions")
    else:
        if verbose: print("H0: Do not reject the null hypothesis - there is no significant difference between the two distributions")
    return initial_statistic, initial_p_value, hypothesis