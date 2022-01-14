# Search for a parameter space that generates the highest level of bistability
import csv
import datetime
from scipy.integrate import odeint
from scipy.stats import loguniform
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

chunks = 94
# Size of parameter sets to generate and simulate over
size = 250
# Number of times to zoom into max window of previous layer
layers = 2
# Number of windows per start, stop range. Fixed for all layers
iter_num = 4
# Tuple of start, stop ranges for iteration 1.
i1_start_stop = (0.1, 10)


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
    path = "subsetResults.csv"
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


def window_gen(spec_param, size, start, stop, num):
    """
    spec_param: parameter to focus analysis of
    size: number of parameter sets per window
    start: beginning of logspace of multipliers to iterate through
    stop: end of logspace of multipliers to iterate through
    num: number of windows to iterate through (odd, preferrably)

    return: list of dataframes size of num - 1
    """
    # Generate non-focus parameters in loguniform
    # Note: these are held the same for each window
    base = {}
    for param in param_names:
        if param != spec_param:
            base[param] = []
            base[param].extend(np.array(loguniform.rvs(0.1, 10, size=size)) * params[param])
    # Logspace of iteration range
    steps = np.logspace(np.log10(start), np.log10(stop), num=num)
    generated_dfs = []

    # Generate parameters in uniform for each window
    for i in range(len(steps)-1):
        sub_gen = base.copy()
        sub_gen[spec_param] = []
        sub_gen[spec_param].extend(np.random.uniform(steps[i], steps[i+1], size=size) * params[spec_param])
        sub_df = pd.DataFrame(sub_gen)
        generated_dfs.append(pd.DataFrame(sub_gen))

    return generated_dfs, steps


# Save figure of bistable rates with x-axis of window midpoint
def plot_rates(midpts, bis_rates, param, save_dir):
    # Sort the values to display
    plot_dict = {"midpt": midpts, "bis_rates": bis_rates}
    plot_axes = pd.DataFrame(plot_dict).sort_values("midpt")
    midpts = plot_axes["midpt"].tolist(); bis_rates = plot_axes["bis_rates"].tolist()

    plt.figure()
    plt.plot(midpts, bis_rates)
    plt.title(f"{param} Sweeping")
    plt.ylabel("Bistable Rate")
    plt.xlabel("Window Stop")
    plt.xscale("log")
    plt.tight_layout()
    plt.savefig(f"{save_dir + param}.jpg", transparent=False)


# Save csv of bistable rates and window
def tabulate_window(steps, bis_rates, param, save_dir):
    windows_dict = {
        "start": steps[:-1],
        "stop": steps[1:],
        "rate": bis_rates
        }

    window_df = pd.DataFrame(windows_dict)
    window_df.to_csv(f"{save_dir + param}.csv")


for param in param_names:
    # Initialize iteration 1 starts and stops for each parameter
    start, stop = i1_start_stop

    # Store all layers' data to plot later
    cumulative_midpts = []
    cumulative_brs = []
    
    for l in range(layers):
        save_dir = f"./SweepSearching/iteration{l}/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        layer_rates = []
        # Generate parameter sets and chunk for parallel processing
        dfs, steps = window_gen(param, size=size, start=start, stop=stop, num=iter_num)
        for df in dfs:
            df_chunks = df_chunker(df, chunks=chunks)
            bis_rate = Parallel(n_jobs=-1)(delayed(run_sim)(sub_df) for sub_df in df_chunks)
            bis_rate = np.sum(np.array(bis_rate))/size
            cumulative_brs.append(bis_rate); layer_rates.append(bis_rate)

        # Calculate midpoints
        midpts = [np.mean([steps[i], steps[i+1]]) for i in range(len(steps)-1)]
        cumulative_midpts.extend(midpts)

        # Index of max rate
        i_max = np.argmax(layer_rates)
        # Determine and update next iteration's start and stop
        start = steps[i_max]; stop = steps[i_max+1]

        # Record data
        tabulate_window(steps, layer_rates, param, save_dir)

    # plot, save cumulated results
    plot_rates(cumulative_midpts, cumulative_brs, param, save_dir)
