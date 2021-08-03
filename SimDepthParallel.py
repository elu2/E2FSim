import csv
import datetime
from scipy.integrate import odeint
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import os

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
    "k_P": 45,
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
    "K_P": 2
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
    
    
# Takes the steady states of E2F on and off conditions and serum concentrations
# Returns distance between on and off IC curves and the [S]s of change
# If bistable resettability is not achieved, return None.

def delta_dist(EE_SS_on, EE_SS_off, serum_con):
    bc_on = []
    bc_off = []
    
    i = 0
    for i in range(len(EE_SS_on) - 1):
        if EE_SS_on[i + 1] - EE_SS_on[i] > 0.1:
            bc_on.append(i)

        if EE_SS_off[i + 1] - EE_SS_off[i] > 0.1:
            bc_off.append(i)
            break
    
    if len(bc_on) == 0 or len(bc_off) == 0:
        return None
    
    base_on = serum_con[bc_on[0]]
    base_off = serum_con[bc_off[0]]
            
    diff = abs(base_on - base_off)
    return [diff, base_on, base_off]


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
    
    k_P_prime = k_P / (K_P + I)
    
    dMdt = mm(k_M, S, K_S, S) - (d_M * M)
    dCDdt = mm(k_CD, M, K_M, M) + mm(k_CDS, S, K_S, S) - d_CD * CD
    dCEdt = mm(k_CE, E, K_E, E) - (d_CE * CE)
    dEdt = mm(k_E, M, K_M, M) * mm(1, E, K_E, E) + mm(k_b, M, K_M, M) + mm(k_P_prime, CD * RE, K_CD, RE) + mm(k_P_prime, CE * RE, K_CE, RE) - (d_E * E) - (k_RE * R * E)
    dRdt = k_R + mm(k_DP, RP, K_RP, RP) - (k_RE * R * E) - mm(k_P_prime, CD * R, K_CD, R) - mm(k_P_prime, CE * R, K_CE, R) - (d_R * R)
    dRPdt = mm(k_P_prime, CD * R, K_CD, R) + mm(k_P_prime, CE * R, K_CE, R) + mm(k_P_prime, CD * RE, K_CD, RE) + mm(k_P_prime, CE * RE, K_CE, RE) - mm(k_DP, RP, K_RP, RP) - (d_RP * RP)
    dREdt = (k_RE * R * E) - mm(k_P_prime, CD * RE, K_CD, RE) - mm(k_P_prime, CE * RE, K_CE, RE) - (d_RE * RE)
    dIdt = k_I - (d_I * I)
        
    return [dMdt, dCDdt, dCEdt, dEdt, dRdt, dRPdt, dREdt, dIdt]


# Chunk parameters for parallel processing

def df_chunker(full_df, chunks):
    dfs = list()
    interval_size = full_df.shape[0]//chunks
    dfs.append(full_df.iloc[0:interval_size, :])

    for i in range(chunks - 1):
        dfs.append(full_df.iloc[(interval_size * (i + 1)):(interval_size * (i + 2)), :])

    if params.depth_params[0] % chunks != 0:
        dfs.append(full_df.iloc[interval_size * chunks: , :])

    return dfs


def run_sim(param_subset):
    for i in range(param_subset.shape[0]):
        globals().update(param_subset.iloc[i].to_dict())

        set_dict = param_subset.iloc[i].to_dict()
        row_vals = list(set_dict.values())

        EE_SS_on = []
        EE_SS_off = []

        for S in serum_con:
            psol = odeint(systems, X0_on, t, args=(S,))
            qsol = odeint(systems, X0_off, t, args=(S,))

            EE_SS_on.append(psol[-1, 3])
            EE_SS_off.append(qsol[-1, 3])

        try:
            dd = [round(x, 4) for x in delta_dist(EE_SS_on, EE_SS_off, serum_con)]
        except TypeError:
            dd = [None, None, None]

        row_vals.extend(dd)
        
        with open("depthRun.csv", 'a+', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(row_vals)


# Time steps
hours = 200
t = np.linspace(0, hours, num=100)

# initial conditions
X0_off = [0, 0, 0, 0, 0, 0, .55, .5]

# Load base parameters for E2F on initial conditions
globals().update(params)
X0_on = list(odeint(systems, X0_off, t, args=(20,))[-1])

# Serum levels
serum_con = np.linspace(0.02, 20, 1000)

with open("runs.log", "a") as log:
    log.write(f"{datetime.datetime.now()}, running depth analysis.\n")

depth_params =  pd.read_csv("depthParameters.csv")
dfs = df_chunker(depth_params, 26)
Parallel(n_jobs=-1)(delayed(run_sim)(sub_df) for sub_df in dfs)

with open("runs.log", "a") as log:
    log.write(f"{datetime.datetime.now()}, running depth analysis completed.\n")
