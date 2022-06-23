# DR Analaysis: get fold changes and perturbation point of two-fold change for each seeding parameter set
# Required files: seed_set.csv, DRXXX.csv from depthRuns

import pandas as pd
import numpy as np
import os
from tqdm import tqdm


def get_subset(dr_df, param):
    scalars = np.logspace(np.log10(0.1), np.log10(10), num=100)
    scalars = pd.DataFrame(pd.Series(scalars)).rename(columns={0:"pert"})
    
    # Add another col in next iteration: stable
    data_cols = ["switch", "bistable", "resettable", "on_thresh", "off_thresh", "d_thresh", "off_SS"]
    dr_df = dr_df[dr_df.an_type == param]
    dr_df = dr_df.sort_values(param)[[param] + data_cols].reset_index(drop=True)
    
    assert dr_df.shape[0] == 100, f"Only {dr_df.shape[0]} simulations for {param}"
    
    dr_df = pd.concat([scalars, dr_df], axis=1)
    
    lower_df = dr_df.iloc[:50]
    upper_df = dr_df.iloc[50:]
    
    return lower_df, upper_df


def get_fc(df, direction, seed_th):
    met_crit = df[(df.switch == True) * (df.bistable == True) * (df.resettable == True)]
    if len(met_crit) == 0:
        return 1, 0
    if direction == "lower":
        num_qual = met_crit.shape[0]
        fc = met_crit.off_thresh.iloc[0]/seed_th
        
    if direction == "upper":
        num_qual = met_crit.shape[0]
        fc = met_crit.off_thresh.iloc[-1]/seed_th
    
    return fc, num_qual


def get_2f_change(df, direction, seed_th):
    # Get direction of dataframe
    met_crit = df[(df.switch == True) * (df.bistable == True) * (df.resettable == True)]
    if len(met_crit) == 0:
        return None

    if (met_crit["off_thresh"].iloc[-1] - met_crit["off_thresh"].iloc[0]) == 0:
        return None

    # Determine if values are increasing in the direction of perturbation
    if direction == "lower":
        increasing = (met_crit["off_thresh"].iloc[-1] - met_crit["off_thresh"].iloc[0]) < 0
    elif direction == "upper":
        increasing = (met_crit["off_thresh"].iloc[-1] - met_crit["off_thresh"].iloc[0]) > 0

    # Determine the expected 2-fold threshold
    if increasing:
        target = 2 * seed_th
    elif not increasing:
        target = seed_th / 2

    if direction == "upper":
        if increasing:
            val_inds = np.where(met_crit["off_thresh"] > target)[0]
        elif not increasing:
            val_inds = np.where(met_crit["off_thresh"] < target)[0]
        if len(val_inds) == 0:
            return None
        else:
            twof_pt = met_crit["pert"].iloc[min(val_inds)]
    if direction == "lower":
        if increasing:
            val_inds = np.where(met_crit["off_thresh"] > target)[0]
        elif not increasing:
            val_inds = np.where(met_crit["off_thresh"] < target)[0]
        if len(val_inds) == 0:
            return None
        else:
            twof_pt = met_crit["pert"].iloc[max(val_inds)]

    return twof_pt


def dr_analysis(dr_num, seed_set_ths, path="./depthRuns/"):
    param_names = ['k_E', 'k_M', 'k_CD', 'k_CDS', 'k_R', 'k_RE', 'k_b', 'k_CE', 'k_I', 'k_P1', 'k_P2',
     'k_DP', 'd_M', 'd_E', 'd_CD', 'd_CE', 'd_R', 'd_RP', 'd_RE', 'd_I', 'K_S', 'K_M',
     'K_E', 'K_CD', 'K_CE', 'K_RP', 'K_P1', 'K_P2']

    dr_df = pd.read_csv(f"{path}DR{dr_num}.csv")

    # get seeding set's threshold
    seed_th = seed_set_ths[dr_num]

    lower_df, upper_df = get_subset(dr_df, "k_E")
    get_fc(lower_df, "lower", seed_th)

    fc_dict = {}
    pt_dict = {}
    nqual_dict = {}
    for param in param_names:
        lower_df, upper_df = get_subset(dr_df, param)
        lower_fc, lower_nval = get_fc(lower_df, "lower", seed_th)
        upper_fc, upper_nval = get_fc(upper_df, "upper", seed_th)

        lower_pt = get_2f_change(lower_df, "lower", seed_th)
        upper_pt = get_2f_change(upper_df, "upper", seed_th)
        
        nqual_dict[param + "_dec"] = lower_nval
        nqual_dict[param + "_inc"] = upper_nval

        fc_dict[param + "_dec"] = lower_fc
        fc_dict[param + "_inc"] = upper_fc

        pt_dict[param + "_dec"] = lower_pt
        pt_dict[param + "_inc"] = upper_pt
    
    # Fold change from seed set threshold
    fc = pd.Series(fc_dict)
    # Point of reaching the two-fold significant value of seed set threshold
    pt = pd.Series(pt_dict)
    # Number of qualified. If 50, it likely didn't end bistability within the ranges of the perturbation
    nq = pd.Series(nqual_dict)
    
    return fc, pt, nq


seed_set_ths = tuple(pd.read_csv("./seed_sets.csv")["off_thresh"])

fc_df = []
pt_df = []
nq_df = []
for i in tqdm(range(1000)):
    fc, pt, nq = dr_analysis(i, seed_set_ths, path="./depthRuns/")
    fc_df.append(fc)
    pt_df.append(pt)
    nq_df.append(nq)

fc_df = pd.DataFrame.from_records(fc_df)
pt_df = pd.DataFrame.from_records(pt_df)
nq_df = pd.DataFrame.from_records(nq_df)

mean_fc = np.mean(fc_df, axis=0)
mean_fc = pd.DataFrame(mean_fc).reset_index()
mean_fc.columns = ["param", "mean_fc"]
mean_fc.sort_values("mean_fc").to_csv("./mean_fc.csv", index=False)

mean_pt = np.mean(pt_df, axis=0)
mean_pt = pd.DataFrame(mean_pt).reset_index()
mean_pt.columns = ["param", "mean_pt"]
mean_pt.sort_values("mean_pt").to_csv("./mean_pt.csv", index=False)

mean_nq = np.mean(nq_df, axis=0)
mean_nq = pd.DataFrame(mean_nq).reset_index()
mean_nq.columns = ["param", "mean_nq"]
mean_nq.sort_values("mean_nq").to_csv("./mean_nq.csv", index=False)
