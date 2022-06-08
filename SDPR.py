# SimDepthParallel: Runs simulations for quiescence depth analysis.

import csv
import datetime
from scipy.integrate import odeint
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import os
import sys

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
    else:
        return low


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
    dEdt = mm(kKpP1, CD * RE, K_CD, RE) + mm(kKpP2, CE * RE, K_CE, RE) + mm(k_E, M,
                                                                            K_M, M) * mm(1, E, K_E, E) + mm(k_b, M, K_M, M) - (d_E * E) - (k_RE * R * E)
    dRdt = k_R + mm(k_DP, RP, K_RP, RP) - mm(kKpP1, CD * R, K_CD, R) - \
        mm(kKpP2, CE * R, K_CE, R) - (d_R * R) - (k_RE * R * E)
    dRPdt = mm(kKpP1, CD * R, K_CD, R) + mm(kKpP2, CE * R, K_CE, R) + mm(kKpP1, CD * RE,
                                                                         K_CD, RE) + mm(kKpP2, CE * RE, K_CE, RE) - mm(k_DP, RP, K_RP, RP) - (d_RP * RP)
    dREdt = (k_RE * R * E) - mm(kKpP1, CD * RE, K_CD, RE) - \
        mm(kKpP2, CE * RE, K_CE, RE) - (d_RE * RE)
    dIdt = k_I - (d_I * I)

    return [dMdt, dCDdt, dCEdt, dEdt, dRdt, dRPdt, dREdt, dIdt]


# Get index of beginning of a list where a sublist exists
def subfinder(l, sl):
    l = list(l)
    sl = list(sl)
    results = []
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind:ind+sll] == sl:
            results.append(ind)

    return results


def calc_switch(EE_SS_off, serum_con, threshold=0.1):
    d_min_max = max(EE_SS_off) - min(EE_SS_off)
    if d_min_max > threshold:
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
    serum_con = np.array(serum_con)
    lgl_EE_SS = (EE_SS > threshold) * 1
    lgl_EE_SS = np.convolve(lgl_EE_SS, np.ones(3, dtype=int), 'valid')

    thresh_i = subfinder(lgl_EE_SS, np.array([1, 2, 3]))
    if len(thresh_i) > 0:
        hm_con = serum_con[thresh_i[0]]
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
        dfs.append(full_df.iloc[interval_size * chunks:, :])

    return dfs


# param_subset: a dictionary of parameters and analysis focus
# decimals: many calculations depend on operations with a tolerance. Rounding standardizes a tolerance of 1e-{decimal}
# n_retain: retain full outputs of simulations for n DepthParams
def run_sim(param_subset, decimals=3, n_retain=0):
    for i in range(param_subset.shape[0]):
        globals().update(param_subset.iloc[i].to_dict())
        X0_off = list(odeint(systems, X0_init, t, args=(0,))[-1])
        X0_on = list(odeint(systems, X0_off, t, args=(50,))[-1])

        inst_at = an_type
        inst_at_val = globals()[an_type]

        set_dict = param_subset.iloc[i].to_dict()
        row_vals = list(set_dict.values())

        EE_SS_on = []
        EE_SS_off = []        

        # Record full output of values
        if array_index < n_retain:
            if not os.path.exists(f"./retainedData/DR{array_index}/"):
                os.mkdirs(f"./retainedData/DR{array_index}/")
        
        # Run simulation
        for S in serum_con:
            psol = odeint(systems, X0_on, t, args=(S,))
            qsol = odeint(systems, X0_off, t, args=(S,))

            EE_SS_on.append(psol[-1, 3])
            EE_SS_off.append(qsol[-1, 3])
            
            if array_index < n_retain:
                retain_df = pd.DataFrame({"on": psol[:, 3], "off", qsol[:, 3]})
                retain_df.to_csv(f"./retainedData/DR{array_index}/{inst_at}-{inst_at_val}.csv", index=False)
        
        EE_SS_on = np.around(EE_SS_on, decimals)
        EE_SS_off = np.around(EE_SS_off, decimals)
        
        off_SS = EE_SS_off[-1]

        # Calculate properties of the system
        switch = calc_switch(EE_SS_off, serum_con)
        resettable = calc_resettable(EE_SS_off, EE_SS_on)

        # Calculate the thresholds of activation/deactivation
        hm_off, hm_on, dhm = act_deact(EE_SS_off, EE_SS_on, serum_con)

        if dhm is not None:
            bistable = dhm > 0.2
        else:
            bistable = False

        if hm_off is not None:
            sound = (hm_off >= 0.5) & (hm_off <= 10)
        else:
            sound = False

        row_vals.extend([switch, bistable, resettable,
                        sound, hm_on, hm_off, dhm, off_SS])

        with open(f"./depthRuns/DR{array_index}.csv", 'a+', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(row_vals)


# Find nearest value in a list to a scalar
def nearest_val(in_arr, val):
    arr_diff = abs(np.array(in_arr) - val)
    min_vals = np.where(arr_diff == min(arr_diff))
    return(min_vals[0][-1])


# returns off half-point, on half-point, and difference in half-points in this order.
def act_deact(EE_SS_off, EE_SS_on, serum_con, tolerance=0):
    EE_SS_on = np.array(EE_SS_on)
    EE_SS_off = np.array(EE_SS_off)
    donoff = EE_SS_on - EE_SS_off

    # Logical vector for values where differences in trajectories are greater than tolerance
    lgl_tol = list((abs(donoff) > tolerance) * 1)

    # If neither curves separate from each other, return none for all
    if sum(lgl_tol) == 0:
        return [None, None, None]

    # Get first and last indices of a bistable region
    lgl_tol = [loc for loc, val in enumerate(lgl_tol) if val == 1]
    min_i = min(lgl_tol)
    max_i = max(lgl_tol)

    # Handles if both trajectories start off different
    if min_i == 0:
        min_i = 1

    if max_i == len(serum_con) - 1:
        # In this case, the on initial condition trajectory did not turn on, but off did turn on
        if donoff[max_i] < 0:
            off_bis = EE_SS_off[min_i - 1: max_i + 2]
            off_hv = (off_bis[0] + off_bis[-1]) / 2
            nrst_off = nearest_val(off_bis, off_hv)
            off_bis_hm = serum_con[min_i - 1 + nrst_off]

            return [off_bis_hm, None, None]

        # Off initial condition trajectory did not turn on, but on did
        elif donoff[max_i] > 0:
            on_bis = EE_SS_on[min_i - 1: max_i + 1]
            on_hv = (on_bis[0] + on_bis[-1]) / 2
            nrst_on = nearest_val(on_bis, on_hv)
            on_bis_hm = serum_con[min_i - 1 + nrst_on]

            return [None, on_bis_hm, None]

    # Otherwise normally assess half-range point of bistable region
    off_bis = EE_SS_off[min_i - 1: max_i + 2]
    off_hv = (off_bis[0] + off_bis[-1]) / 2
    nrst_off = nearest_val(off_bis, off_hv)
    off_bis_hm = serum_con[min_i - 1 + nrst_off]

    on_bis = EE_SS_on[min_i - 1: max_i + 1]
    on_hv = (on_bis[0] + on_bis[-1]) / 2
    nrst_on = nearest_val(on_bis, on_hv)
    on_bis_hm = serum_con[min_i - 1 + nrst_on]

    return [off_bis_hm, on_bis_hm, abs(on_bis_hm - off_bis_hm)]


def powspace(start, stop, power, num):
    start = np.power(start, 1/float(power))
    stop = np.power(stop, 1/float(power))
    return np.power(np.linspace(start, stop, num=num), power)


# Time steps
t = powspace(0, 1000, 4, 100)

# initial conditions
X0_init = [0, 0, 0, 0, 0, 0, .55, .5]

# Load base parameters for E2F on initial conditions
seed_params = pd.read_csv("seed_sets.csv").iloc[int(array_index)]
globals().update(seed_params)

# Serum levels
serum_con = np.logspace(np.log10(0.01), np.log10(50), 500)

depth_params = pd.read_csv(f"./depthParams/DP{array_index}.csv")

dfs = df_chunker(depth_params, 180)

Parallel(n_jobs=-1)(delayed(run_sim)(sub_df) for sub_df in dfs)
