import pandas as pd
import numpy as np

reps = 2000
lower_oom = 0.01
upper_oom = 100

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

params = {
    "k_E": [],
    "k_M": [],
    "k_CD": [],
    "k_CDS": [],
    "k_R": [],
    "k_RE": [],
    "k_b": [],
    "k_CE": [],
    "k_I": [],
    "k_P": [],
    "k_DP": [],
    "d_M": [],
    "d_E": [],
    "d_CD": [],
    "d_CE": [],
    "d_R": [],
    "d_RP": [],
    "d_RE": [],
    "d_I": [],
    "K_S": [],
    "K_M": [],
    "K_E": [],
    "K_CD": [],
    "K_CE": [],
    "K_RP": [],
    "K_P": []
}


scalars = np.linspace(lower_oom, upper_oom, num=reps)

i = 0
for param_key in params.keys():
    for base_key in base.keys():
        if base_key == param_key:
            scaled_vec = list(scalars * base[base_key])
            params[base_key] = scaled_vec
            
        else:
            params[base_key] = list(np.full(reps, base[base_key]))
    
    pd.DataFrame(params).to_csv(f"./depthLib/{param_key}{i}depth.csv", index=False)
    i += 1
