# SimDepthParallel: Runs simulations for quiescence depth analysis.

import csv
import datetime
from scipy.integrate import odeint
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import os, sys

array_index = sys.argv[1]

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


# Michaelis-Menten template

def mm(num_k, num_con, denom_K, denom_con):
    val = (num_k * num_con) / (denom_K + denom_con)

    return val


# Behavior of serum concentration. Vestigial.

def serum(time):
    R = 100000
    hi = 20
    low = .5
    if time < R:
        return hi
    else: return low


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


# Get index of beginning of a list where a sublist exists
def subfinder(in_list, pattern):
    in_list = np.array(in_list); pattern = np.array(pattern)
    matches = []
    for i in range(len(in_list)):
        if in_list[i] == pattern[0] and (in_list[i:i+len(pattern)] == pattern).all():
            matches.append(i)
    return matches


def calc_switch(EE_SS_off, serum_con, threshold=0.1):
    d_min_max = max(EE_SS_off) - min(EE_SS_off)
    if d_min_max > threshold:
        return True
    return False


# If a certain number (n or n//2) of concentrations have a certain property, it is bistable
def calc_bistable(EE_SS_off, EE_SS_on):
    n = (len(EE_SS_off) // 25) * 2
    
    EE_SS_off = np.array(EE_SS_off); EE_SS_on = np.array(EE_SS_on)
    delta_on_off = EE_SS_on - EE_SS_off
    
    thresh1_dmm = (max(EE_SS_off) - min(EE_SS_off)) * 0.1
    thresh2_dmm = (max(EE_SS_off) - min(EE_SS_off)) * 0.2
    
    con1 = (delta_on_off >= thresh1_dmm) * 1
    con1 = np.convolve(con1, np.ones(n, dtype=int), 'valid')
    
    con2 = (delta_on_off >= thresh2_dmm) * 1
    con2 = np.convolve(con2, np.ones(n//2, dtype=int), 'valid')
    
    # If n consecutive reaching the criteria, bistability exists
    if n in con1 or n//2 in con2:
        return True
    return False


def calc_resettable(EE_SS_off, EE_SS_on):
    t0_delta = EE_SS_off[0] - EE_SS_on[0]
    if abs(t0_delta) < 0.001:
        return True
    return False


# Find serum concentration where a EE_SS first reaches 0.1

def find_halfmax(EE_SS, serum_con, threshold=0.1):
    EE_SS = np.array(EE_SS)
    lgl_EE_SS = (EE_SS > threshold) * 1
    lgl_EE_SS = np.convolve(lgl_EE_SS, np.ones(3, dtype=int), 'valid')

    thresh_i = subfinder(lgl_EE_SS, np.array([1, 2, 3]))
    if len(thresh_i) > 0:
        hm_con = serum_con[thresh_i[0]][0]
    else:
        hm_con = None
    
    return hm_con


# Chunk parameters for parallel processing

def df_chunker(full_df, chunks):
    dfs = list()
    interval_size = full_df.shape[0]//chunks
    dfs.append(full_df.iloc[0:interval_size, :])

    for i in range(chunks - 1):
        dfs.append(full_df.iloc[(interval_size * (i + 1)):(interval_size * (i + 2)), :])

    if depth_params.shape[0] % chunks != 0:
        dfs.append(full_df.iloc[interval_size * chunks: , :])

    return dfs


def run_sim(param_subset):
    for i in range(param_subset.shape[0]):
        globals().update(param_subset.iloc[i].to_dict())
        inst_at = an_type
        inst_at_val = globals()[an_type]

        set_dict = param_subset.iloc[i].to_dict()
        row_vals = list(set_dict.values())

        EE_SS_on = []
        EE_SS_off = []

        # Run simulation
        for S in serum_con:
            psol = odeint(systems, X0_on, t, args=(S,))
            qsol = odeint(systems, X0_off, t, args=(S,))

            EE_SS_on.append(psol[-1, 3])
            EE_SS_off.append(qsol[-1, 3])
        
        # Calculate properties of the system
        switch = calc_switch(EE_SS_off, serum_con)
        bistable = calc_bistable(EE_SS_off, EE_SS_on)
        resettable = calc_resettable(EE_SS_off, EE_SS_on)

        # Calculate the thresholds of activation/deactivation
        hm_off = find_halfmax(EE_SS_off, serum_con)
        hm_on = find_halfmax(EE_SS_on, serum_con)
        if hm_off == None or hm_on == None:
            dhm = None
        else:
            dhm = hm_off - hm_on
        
        row_vals.extend([switch, bistable, resettable, hm_on, hm_off, dhm])

        with open(f"./depthRuns/DR{array_index}.csv", 'a+', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(row_vals)
        
        
# Time steps
hours = 200
t = np.linspace(0, hours, num=100)

# initial conditions
X0_off = [0, 0, 0, 0, 0, 0, .55, .5]
 
# Load base parameters for E2F on initial conditions
seed_params = pd.read_csv("seed_sets.csv").iloc[array_index]
globals().update(seed_params)
X0_on = list(odeint(systems, X0_off, t, args=(20,))[-1])

# Serum levels
serum_con = np.logspace(np.log10(0.01), np.log10(20), 500)

depth_params =  pd.read_csv(f"./depthParams/DP{array_index}.csv")

dfs = df_chunker(depth_params, 94)

Parallel(n_jobs=-1)(delayed(run_sim)(sub_df) for sub_df in dfs)
