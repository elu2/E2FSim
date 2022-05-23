# Short script to filter non-bistable results from previous step
# Prunes according to trunc_upper

import pandas as pd

trunc_upper = 5000

psr = pd.read_csv("./pre_seed_results.csv")

psr = psr[psr["bistable"] == True].reset_index(drop=True)
trunc = min(psr.shape[0], trunc_upper)

psr = psr.drop(["switch", "bistable", "resettable", "on_thresh", "off_thresh", "d_thresh"], axis=1)
psr = psr[:trunc]

psr.to_csv("seed_sets.csv", index=False)
