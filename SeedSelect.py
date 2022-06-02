# Short script to filter non-bistable results from previous step
# Prunes according to trunc_upper

import pandas as pd

trunc_upper = 5000

psr = pd.read_csv("./pre_seed_results.csv")

psr = psr[(psr["bistable"] == True) & (psr["switch"] == True) & (
    psr["resettable"] == True) & (psr["sound"] == True)].reset_index(drop=True)
trunc = min(psr.shape[0], trunc_upper)

psr = psr.drop(["switch", "bistable", "resettable", "sound",
               "on_thresh", "d_thresh", "off_SS"], axis=1)
psr = psr[:trunc]

psr.to_csv("seed_sets.csv", index=False)
