import numpy as np
import pandas as pd


def lognuniform(low=0, high=1, size=None, base=10):
    return np.power(base, np.random.uniform(low, high, size))


# Initialize empty file for parameter sets when analyzing bistability
col_names = ['k_E', 'k_M', 'k_CD', 'k_CDS', 'k_R', 'k_RE', 'k_b', 'k_CE', 'k_I', 'k_P1', 'k_P2', 'k_DP', 'd_M', 'd_E', 'd_CD', 'd_CE', 'd_R', 'd_RP', 'd_RE', 'd_I', 'K_S', 'K_M', 'K_E', 'K_CD', 'K_CE', 'K_RP', 'K_P1', 'K_P2',
             'switch', 'bistable', 'resettable', 'sound', 'on_thresh', 'off_thresh', 'd_thresh', 'off_SS', 'stable']
pd.DataFrame(columns=col_names).to_csv(f"./pre_seed_results.csv", index=False)


# Generate parameters for seed sets (loguniform)
n_params = 100000
lower_oom = 0.1
upper_oom = 10

param_names = ['k_E', 'k_M', 'k_CD', 'k_CDS', 'k_R', 'k_RE', 'k_b', 'k_CE', 'k_I', 'k_P1', 'k_P2', 'k_DP', 'd_M',
               'd_E', 'd_CD', 'd_CE', 'd_R', 'd_RP', 'd_RE', 'd_I', 'K_S', 'K_M', 'K_E', 'K_CD', 'K_CE', 'K_RP', 'K_P1', 'K_P2']

base = {
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

params = {}
# Initialize params with empty list
for name in param_names:
    params[name] = []

for name in param_names:
    params[name].extend(list(np.around(lognuniform(np.log10(
        base[name] * lower_oom), np.log10(base[name] * upper_oom), size=n_params, base=10), 4)))

pd.DataFrame(params).to_csv("pre_seed_sets.csv", index=False)
