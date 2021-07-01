#!/usr/bin/env python
# coding: utf-8

import csv
import datetime
from scipy.integrate import odeint
from joblib import Parallel, delayed
import numpy as np
import pandas as pd


def model_name(model):
    model_name = ""
    for state in model:
        model_name += str(state)
    return "SM" + model_name + ".csv"


# Positive Michaelis-Menten equation
# state: 0 or 1 depending on if a model component will be on or off
# a_or_m: return additive or multiplicative identity when model component is off

def mm_pos(A_n, K_n, n_n, a_or_m, state):
    if state == 0:
        if a_or_m == "a":
            return 0
        if a_or_m == "m":
            return 1
        
    else:
        if A_n < 0:
            A_n = 0

        if A_n >= 11:
            A_n = 11
            
        val = A_n**n_n/(K_n**n_n + A_n**n_n)
        return val


# Negative Michaelis-Menten equation
# state: 0 or 1 depending on if a model component will be on or off
# a_or_m: return additive or multiplicative identity when model component is off

def mm_neg(A_n, K_n, n_n, a_or_m, state):
    if state == 0:
        if a_or_m == "a":
            return 0
        if a_or_m == "m":
            return 1
        
    else:
        if A_n < 0:
            A_n = 0

        if A_n >= 11:
            A_n = 11

        val = K_n**n_n/(K_n**n_n + A_n**n_n)
        return val

    
# For the special 9th model component link
# state7: link 7 on or off
# state9: link 9 on or off

def link_9(state7, state9, MD, EE, n_7, n_9, K_7, K_9, beta_EE=None):
    if state9 == 0:
        link7 = mm_pos(MD, K_7, n_7, "m", state7)
        return link7
    
    if state9 == 1:
        link9a = mm_pos(MD, K_7, n_7, "m", state7) * mm_pos(EE, K_9, n_9, "m", 1)
        return link9a
    
    if state9 == 2:
        link9b = mm_pos(MD, K_7, n_7, "m", state7) + beta_EE*mm_pos(EE, K_9, n_9, "a", 1)
        return link9b


# Chunks parameters for parallel processing

def df_chunker(full_df, chunks):
    dfs = []
    interval_size = full_df.shape[0]//chunks
    dfs.append(full_df.iloc[0:interval_size, :])

    for i in range(chunks - 1):
        dfs.append(full_df.iloc[(interval_size * (i + 1)):(interval_size * (i + 2)), :])

    if params.shape[0] % chunks != 0:
        dfs.append(full_df.iloc[interval_size * chunks: , :])

    return dfs


# Performs analysis on switch/bistability behavior.
# Need to implement resettability condition in the future.
# EE_SS_on, EE_SS_off: proliferative and quiescent odeint outputs, respectively.

def cond_analysis(EE_SS_on, EE_SS_off):
    lmda = 0.1
    # Calculate difference between max and min of SS for EE-off initial condition
    EE_min_max = max(EE_SS_off) - min(EE_SS_off)

    # Switch conditions
    switch = False
    if EE_min_max > lmda:
        switch = True

    # Keep record of delta EE_SS
    delta_EE_SS = []
    for SS_off, SS_on in zip(EE_SS_off, EE_SS_on):
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


# Input function for odeint
# Includes logic for upper bound being 11
# Logic for lower bound being 0 should/needs to be implemented

def systems(X, t, S, states):
    EE = X[0]
    MD = X[1]
    RP = X[2]


    dEEdt = (1/tau_EE) * (link_9(states[5], states[7], MD, EE, n_7, n_9, K_7, K_9, beta_EE) * mm_neg(RP, K_6, n_6, "m", states[4]) * mm_neg(MD, K_8, n_8, "m", states[6]) * mm_neg(EE, K_10, n_10, "m", states[8]) - EE)
    dMDdt = (1/tau_MD) * (mm_pos(S, K_1, n_1, "m", 1) + beta_MD*mm_pos(EE, K_2, n_2, "a", states[0]) - MD)
    dRPdt = (1/tau_RP) * ((1 + beta_RP*mm_pos(EE, K_4, n_4, "a", states[2])) * mm_neg(MD, K_3, n_3, "m", states[1]) * mm_neg(EE, K_5, n_5, "m", states[3]) - RP)

    return [dEEdt, dMDdt, dRPdt]


def run_sim(param_subset):
    # Loop through parameters and record steady-state concentrations
    for i in range(param_subset.shape[0]):
        # Update parameters from row of df
        globals().update(param_subset.iloc[i].to_dict())
        
        set_dict = param_subset.iloc[i]
        row_vals = list(set_dict.values())

        EE_SS_on = []
        EE_SS_off = []

        for S in serum_con:
            # Calculate ODEs
            psol = odeint(systems, X0_on, t, args=(S, states))
            qsol = odeint(systems, X0_off, t, args=(S, states))

            # Per-parameter-set steady state recording
            EE_SS_on.append(psol[-1, 0])
            EE_SS_off.append(qsol[-1, 0])

        rebi_bool, bistable_bool = cond_analysis(EE_SS_on, EE_SS_off)

        # Note that the order is bistable -> rebi
        if bistable_bool == 1:
            row_vals.append(1)
        else:
            row_vals.append(0)

        if rebi_bool == 1:
            row_vals.append(1)
        else:
            row_vals.append(0)
            
        with open(model_name(), 'a+', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(row_vals)

        

# Parallelized simulation running

def run_parallel(params, chunks):
    dfs = df_chunker(params, chunks)
    results = Parallel(n_jobs=-1)(delayed(run_sim)(sub_df) for sub_df in dfs)
    

if __name__ == "__main__":
    # --- Loading parameters, states, initial conditions, time steps, and serum concentrations ---
    # Load parameters
    params = pd.read_csv("parameters.csv")

    # Initial conditions
    X0_on = [11, 11, 0.01] # EE-on initial condition
    X0_off = [0.01, 0.01, 11] # EE-off initial condition

    # Time steps
    t = np.linspace(.2, 200, 100)

    # .01 to 20 serum concentration
    serum_con = np.logspace(-2, 1.3, 25)

    # On/off components of model
    # link 9: 0 for off, 1 for 9a, 2 for 9b. The rest are 0 for off, 1 for on
    # Most robust: states = [1, 1, 0, 1, 1, 1, 0, 2, 0]
    # Minimal: states = [0, 1, 0, 1, 1, 1, 0, 0, 0]
    # Links:  2  3  4  5  6  7  8  9  10
    states = [1, 1, 1, 1, 1, 1, 1, 2, 1]
    
    # Log start of run
    with open("runs.log", "a") as log:
        log.write(f"{datetime.datetime.now()}, running single model {states} in {chunks} chunks.\n")
    
    run_parallel(params, chunks=100)
    
    # Log start of run
    with open("runs.log", "a") as log:
        log.write(f"{datetime.datetime.now()}, running single model {states} in {chunks} chunks completed.\n")
