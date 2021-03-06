# QuiescenceDepthInitRecursive: Initializes DR{i}.csv files which SDPR.py writes into.
# Warning: will overwrite existing files. Could lose hours of work.

import pandas as pd
import os

n = 5000

dir_name = "./depthRuns/"

colnames = ['k_E', 'k_M', 'k_CD', 'k_CDS', 'k_R', 'k_RE', 'k_b', 'k_CE', 'k_I', 'k_P1', 'k_P2', 'k_DP', 'd_M', 'd_E', 'd_CD', 'd_CE', 'd_R', 'd_RP', 'd_RE', 'd_I', 'K_S', 'K_M', 'K_E', 'K_CD', 'K_CE', 'K_RP', 'K_P1', 'K_P2', "an_type",
            "switch", "bistable", "resettable", "sound", "on_thresh", "off_thresh", "d_thresh", "off_SS", "stable"]

if not os.path.exists(dir_name):
    os.makedirs(dir_name)

os.chdir(dir_name)

init_df = pd.DataFrame(columns=colnames)

for i in range(n):
    init_df.to_csv(f"DR{i}.csv", index=False)
