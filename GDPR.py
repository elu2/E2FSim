# Read from GDPR.log
# Generate params like GDP.py except from different seeding.

import pandas as pd
import os

# Check if log is missing.
if not os.path.exists(f"GDPR.log"):
    with open("GDPR.log", "w") as file:
        file.write(str(0))

index = None
with open("GDPR.log", "r") as file:
    index = int(file.readline())
    
bistable_sets = pd.read_csv("bistableResults2017.csv")
base = bistable_sets.iloc[index, :-2].to_dict()

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

scalars = np.logspace(np.log10(lower_oom), np.log10(upper_oom), num=reps)

for param_key in list(params.keys())[:-1]:
    params["an_type"].extend([param_key] * reps)
    for base_key in base.keys():
        if base_key == param_key:
            scaled_vec = list(scalars * base[base_key])
            params[base_key].extend(scaled_vec)
            
        else:
            params[base_key].extend(list(np.full(reps, base[base_key])))
    
pd.DataFrame(params).to_csv(f"depthParameters{index}.csv", index=False)

with open("GDPR.log", "w") as file:
    file.write(str(index + 1))
