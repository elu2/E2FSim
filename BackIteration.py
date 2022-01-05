import csv
import datetime
from scipy.integrate import odeint
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import os

# Path to write iteration results out to. Make directory if doesn't already exist.
write_path = "./iterationResults/"
if not os.path.exists(write_path):
    os.makedirs(write_path)

# Number of parameter sets to generate and simulate over
size = 500

# Number of chunks to split parameter set into for parallel processing.
chunks = 94

# Baseline parameter values from 2017 paper.
params = {
    "k_E": 0.4,
    "k_M": 1.0,
    "k_CD": 0.03,
    "k_CDS": 0.45,
    "k_R": 0.18,
    "k_RE": 180,
    "k_b": 0.003,
    "k_CE": 0.35,
    "k_I": 0.15,
    "k_P1": 45,
    "k_P2": 45,
    "k_DP": 3.6,
    "d_M": 0.7,
    "d_E": 0.25,
    "d_CD": 1.5,
    "d_CE": 1.5,
    "d_R": 0.06,
    "d_RP": 0.06,
    "d_RE": 0.03,
    "d_I": 0.3,
    "K_S": 0.5,
    "K_M": 0.15,
    "K_E": 0.15,
    "K_CD": 0.92,
    "K_CE": 0.92,
    "K_RP": 0.01,
    "K_P1": 2,
    "K_P2": 2
}

param_names = list(params.keys())

# Time steps
hours = 200
t = np.linspace(0, hours, num=100)

# initial conditions
X0_off = [0, 0, 0, 0, 0, 0, .55, .5]

# Serum levels
serum_con = np.linspace(0.02, 20, 100)


# Michaelis-Menten template

def mm(num_k, num_con, denom_K, denom_con):
    val = (num_k * num_con) / (denom_K + denom_con)
    
    return val


# Chunk parameters for parallel processing

def df_chunker(full_df, chunks):
    dfs = list()
    interval_size = full_df.shape[0]//chunks
    dfs.append(full_df.iloc[0:interval_size, :])

    for i in range(chunks - 1):
        dfs.append(full_df.iloc[(interval_size * (i + 1)):(interval_size * (i + 2)), :])

    if full_df.shape[0] % chunks != 0:
        dfs.append(full_df.iloc[interval_size * chunks: , :])

    return dfs


# Performs analysis on switch/bistability behavior.
# Need to implement resettability condition in the future.
# E2F_on, E2F_off: proliferative and quiescent odeint outputs, respectively.

def cond_analysis(E2F_on, E2F_off):
    lmda = 0.1
    # Calculate difference between max and min of SS for EE-off initial condition
    EE_min_max = max(E2F_off) - min(E2F_off)

    # Switch conditions
    switch = False
    if EE_min_max > lmda:
        switch = True

    # Keep record of delta EE_SS
    delta_EE_SS = []
    for SS_off, SS_on in zip(E2F_off, E2F_on):
        delta_EE_SS.append(SS_on - SS_off)
        
    # Bistability conditions
    bistable_bool = False
    if sum([i > EE_min_max * .1 for i in delta_EE_SS]) >= 2 and switch:
        bistable_bool = True
    elif sum([i > EE_min_max * .2 for i in delta_EE_SS]) >= 1 and switch:
        bistable_bool = True
    
    # Resettability conditions
    reset_bool = False
    if delta_EE_SS[0] <= .05:
        reset_bool = True
    
    # Assess if both resettable and bistable
    rebi_bool = 0
    if reset_bool and bistable_bool: 
        rebi_bool = 1
    
    return rebi_bool, int(bistable_bool)


def systems(X, t, S):
    # ODEs as vector elements
    M = X[0]
    CD = X[1]
    CE = X[2]
    E = X[3]
    R = X[4]
    RP = X[5]
    RE = X[6]
    I = X[7]
    
    kKpP1 = k_P1 / (K_P1 + I)
    kKpP2 = k_P2 / (K_P2 + I)
        
    dMdt = mm(k_M, S, K_S, S) - (d_M * M)
    dCDdt = mm(k_CD, M, K_M, M) + mm(k_CDS, S, K_S, S) - d_CD * CD
    dCEdt = mm(k_CE, E, K_E, E) - (d_CE * CE)
    dEdt = mm(kKpP1, CD * RE, K_CD, RE) + mm(kKpP2, CE * RE, K_CE, RE) + mm(k_E, M, K_M, M) * mm(1, E, K_E, E) + mm(k_b, M, K_M, M) - (d_E * E) - (k_RE * R * E)
    dRdt = k_R + mm(k_DP, RP, K_RP, RP) - mm(kKpP1, CD * R, K_CD, R) - mm(kKpP2, CE * R, K_CE, R) - (d_R * R) - (k_RE * R * E)
    dRPdt = mm(kKpP1, CD * R, K_CD, R) + mm(kKpP2, CE * R, K_CE, R) + mm(kKpP1, CD * RE, K_CD, RE) + mm(kKpP2, CE * RE, K_CE, RE) - mm(k_DP, RP, K_RP, RP) - (d_RP * RP)
    dREdt = (k_RE * R * E) - mm(kKpP1, CD * RE, K_CD, RE) - mm(kKpP2, CE * RE, K_CE, RE) - (d_RE * RE)
    dIdt = k_I - (d_I * I)
        
    return [dMdt, dCDdt, dCEdt, dEdt, dRdt, dRPdt, dREdt, dIdt]
    
    
def csv_init(path, from_df_names):
    col_names = list(from_df_names.columns) + ["bistable"]
    pd.DataFrame(columns=col_names).to_csv(path, index=False)
    print(f"Created {path}")


def run_sim(param_subset):
    bis_counter = 0
    for i in range(param_subset.shape[0]):
        globals().update(param_subset.iloc[i].to_dict())
        
        set_dict = param_subset.iloc[i].to_dict()
        row_vals = list(set_dict.values())

        # Load base parameters for E2F on initial conditions
        X0_on = list(odeint(systems, X0_off, t, args=(20,))[-1])

        E2F_on = []
        E2F_off = []

        for S in serum_con:
            # Calculate ODEs
            psol = odeint(systems, X0_on, t, args=(S,))
            qsol = odeint(systems, X0_off, t, args=(S,))

            # Per-parameter-set steady state recording
            E2F_on.append(psol[-1, 3])
            E2F_off.append(qsol[-1, 3])

        rebi_bool, bistable_bool = cond_analysis(E2F_on, E2F_off)
        
        if bistable_bool: 
            bis_counter += 1
    
    return bis_counter
            
            
def zoom_gen(p_params, n_params, unimp, size):
    # Generate parameters with certain narrowed parameter space
    generated = {}
    for param in n_params + p_params + unimp:
        generated[param] = []

    for param in p_params:
        generated[param].extend(np.random.uniform(5, 10, size=size) * params[param])

    for param in n_params:
        generated[param].extend(np.random.uniform(0.1, 0.5, size=size) * params[param])

    for param in unimp:
        generated[param].extend([params[param]]*size)

    generated_df = pd.DataFrame(generated)
    
    return generated_df


# Initialized parameter classes as determined by a prior LASSO fit's coefficients. 28 total.
p_init = ['k_b', 'k_E', 'k_CD', 'k_CDS', 'd_R', 'k_M', 'd_I', 'k_CE', 'k_P1', 'k_P2']
n_init = ['k_RE', 'K_P2', 'd_CE', 'K_CE', 'K_S', 'k_DP', 'K_P1', 'd_RP', 'd_M', 'k_I', 'd_CD', 'K_CD', 'K_RP', 'K_M', 'd_E', 'k_R', 'K_E', 'd_RE']

# Loop through as many parameters as initialized
for iteration in range(len(p_init + n_init) - 1):
    pn_comb = p_init + n_init
    unimp_init = list(set(param_names) - set(p_init) - set(n_init))

    rate_log = []
    for i in range(len(pn_comb)):
        # Re-initialize parameter classes
        p_params = p_init.copy()
        n_params = n_init.copy()
        unimp = unimp_init.copy()

        # Re-class a parameter to unimportant
        reverted = pn_comb[i]
        if reverted in p_params:
            reverted = p_params.pop(p_params.index(reverted))
            unimp.append(reverted)
        elif reverted in n_params:
            reverted = n_params.pop(n_params.index(reverted))
            unimp.append(reverted)

        # Generate parameter sets in a zoomed parameter space corresponding to class
        generated_df = zoom_gen(p_params, n_params, unimp, size=size)

        # Chunk parameters to run in parallel (94 for HPC)
        gen_chunks = df_chunker(generated_df, chunks)
        bis_counts = Parallel(n_jobs=-1)(delayed(run_sim)(sub_df) for sub_df in gen_chunks)

        bis_rate = np.sum(bis_counts)/size

        rate_log.append(bis_rate)

    # Obtain parameters which lowered bistability rate least and most
    max_is = [i for i, value in enumerate(rate_log) if value == np.amax(rate_log)]
    max_params = [pn_comb[i] for i in max_is]

    min_is = [i for i, value in enumerate(rate_log) if value == np.amin(rate_log)]
    min_params = [pn_comb[i] for i in min_is]

    # Construct dataframe of results and export
    param_rates = {
        "param": p_init + n_init,
        "bis_rate": rate_log
    }

    rate_df = pd.DataFrame(param_rates).sort_values("bis_rate")
    rate_df.to_csv(write_path + f"l{iteration}Rates.csv", index=False)

    # Determine next parameter to revert parameter space
    nrev = rate_df.iloc[-1][0]
    nrate = rate_df.iloc[-1][1]

    if nrev in p_init:
        nrev_i = p_init.index(nrev)
        nrev = p_init.pop(nrev_i)
        print(f"Next revert at {nrate} bistable rate: {nrev}")

    elif nrev in n_init:
        nrev_i = n_init.index(nrev)
        nrev = n_init.pop(nrev_i)
        print(f"Next revert at {nrate} bistable rate: {nrev}")
