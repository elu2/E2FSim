#!/usr/bin/env python
# coding: utf-8

from scipy.integrate import odeint
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import time
import itertools as it


# Plot the relevent outputs used downstream.
# input_obj: for type 1, input odeint output. for type 2, just look at the code to figure out input... 
# plot_type: 1 for concentration plots, 2 for EE steady state plots

def plotter(input_obj, plot_type):
    if plot_type == 1:
        EE = input_obj[:, 0]
        MD = input_obj[:, 1]
        RP = input_obj[:, 2]

        plt.plot()
        plt.title('Concentrations Plot')
        plt.xlabel('Time (hrs)')
        plt.ylabel('Concentration (nM/mL)')
        plt.plot(t, EE, label="EE")
        plt.plot(t, MD, label="MD")
        plt.plot(t, RP, label="RP")
        plt.legend()
        plt.show()

    if plot_type == 2:
        plt.plot(input_obj[0][0], input_obj[0][1], label="EE On")
        plt.plot(input_obj[1][0], input_obj[1][1], label="EE Off")
        plt.legend(loc="best")
        plt.ylabel("EE_SS")
        plt.xlabel("log10 [S]")
        plt.show()


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
        # Unsure about applying maximum
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
        # Unsure about applying maximum
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


# Create library of models with all linkers' combinations

def model_library():
    # Start with baseline model with all links off
    models_a = [(0, 0, 0, 0, 0, 0, 0, 0, 0)]
    
    # find permutations of all possible linkers with 9a
    for i in range(9):
        none_on = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        for j in range(i+1):
            none_on[j] = 1
        models_a.extend(list(set(it.permutations(none_on))))

    # Create all possible permutations with 9b linker and add to final list
    models_b = models_a
    models_b = [list(tup) for tup in models_b]

    i = 0
    for i in range(len(models_b)):
        models_b[i][7] = 2

    models_b = [tuple(lst) for lst in models_b]
    models_a.extend(set(models_b))
    
    return models_a


def run_sim(states):
    rebi_count = 0
    bistable_count = 0
    
    # Loop through parameters and record steady-state concentrations
    for i in range(params.shape[0]):
        # Update parameters from row of df
        globals().update(params.iloc[i].to_dict())

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
        rebi_count += rebi_bool
        bistable_count += bistable_bool
        
        print(f"States: {states}, {rebi_count}/{i} bistable and resettable.", end="\r")
    
    return states, bistable_count, rebi_count


# Parallelized simulation running

def run_parallel(params, cpus, run_range):
    lower, upper = run_range
    results = Parallel(n_jobs=cpus)(delayed(run_sim)(model) for model in model_library()[lower:upper])
    
    for result in results:
        row = list(result[0])
        row.append(result[1])
        row.append(result[2])
        pd.DataFrame([row]).to_csv("model_rebi_counts.csv", mode="a", header=False, index=False)


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

    start = time.time()
    rebi = run_parallel(params, -1, (0, 2))
    print(f"Total runtime: {round((time.time() - start)/60, 3)}min")
