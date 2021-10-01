# Processes depth run data into fold changes for LASSO

import pandas as pd
import numpy as np
import os

os.chdir("/xdisk/guangyao/elu2/E2FSim/depthRuns/")
all_files = os.listdir(".")

params = ['K_CE', 'k_P1', 'k_I', 'd_CD', 'k_E', 'd_M', 'k_P2', 'K_M', 'k_RE', 'K_E', 'd_CE', 'k_CE', 'd_E', 'd_I', 'k_DP', 'k_b', 'K_CD', 'K_P1', 'k_M', 'K_S', 'k_CD', 'K_RP', 'd_RP', 'd_R', 'k_R', 'k_CDS', 'K_P2', 'd_RE']

lower_oom = 0.1
upper_oom = 10
reps=100

scalars = np.logspace(np.log10(lower_oom), np.log10(upper_oom), num=reps)
zeros = np.zeros(100)


def fold_change(x, y):
    return (y - x)/x


# df: DRXXXX.csv dataframe
# base_df: compiled overall ranking for a depth run
def fold_changes(df, base_df):    
    for param in params:
        # Initialize a row with zeros
        overall_fc = {}
        for paramj in params:
            overall_fc[paramj] = [0]
        
        param_df = df[df["an_type"]==param].reset_index(drop=True)

        # Make zero columns for all but the current parameter
        zero_cols = params.copy()
        zero_cols.remove(param)
        zero_df = pd.DataFrame({col_name: zeros for col_name in zero_cols})
        param_df[zero_cols] = zero_df
        param_df = param_df.dropna().reset_index(drop=True)

        x1 = param_df[param].iloc[0]
        y1 = param_df[param].iloc[-1]

        x2 = param_df["dOnOff"].iloc[0]
        y2 = param_df["dOnOff"].iloc[-1]

        fc1 = fold_change(x1, y1)
        fc2 = fold_change(x2, y2)
        
        overall_fc[param] = [fc1]
        overall_fc["dOnOff"] = [fc2]
        
        overall_fc_df = pd.DataFrame(overall_fc)
        base_df = base_df.append(overall_fc_df)
    
    return base_df


if __name__ == "__main__":
    base_df = pd.DataFrame()
    for file in all_files:
        df = pd.read_csv(file)
        base_df = fold_changes(df, base_df)
    
    base_df.to_csv(f"../fold_changes.csv", index=False)
