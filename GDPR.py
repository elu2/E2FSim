# Generate perturbed parameter sets from seed_sets.csv

import pandas as pd
import numpy as np
import os

reps = 100
lower_oom = 0.1
upper_oom = 10
n_pert = 5000

scalars = np.logspace(np.log10(lower_oom), np.log10(upper_oom), num=reps)

seed_sets = pd.read_csv("seed_sets.csv")

if not os.path.exists("./depthParams/"):
    os.makedirs("./depthParams/")

for i in range(n_pert):
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
        "k_P1": [],
        "k_P2": [],
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
        "K_P1": [],
        "K_P2": [],
        "an_type": []
    }

    # Index until -1 because last column is off_threshold
    base = seed_sets.iloc[i, :28].to_dict()

    for param_key in list(params.keys())[:-1]:
        params["an_type"].extend([param_key] * reps)
        for base_key in base.keys():
            if base_key == param_key:
                scaled_vec = list(scalars * base[base_key])
                params[base_key].extend(scaled_vec)

            else:
                params[base_key].extend(list(np.full(reps, base[base_key])))

    pd.DataFrame(params).to_csv(f"./depthParams/DP{i}.csv", index=False)
