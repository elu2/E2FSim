import pandas as pd
import numpy as np
import os
from joblib import Parallel, delayed


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
    seed_th = seed_set_ths[0]

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
    fc = pd.DataFrame(pd.Series(fc_dict)).transpose()
    # Point of reaching the two-fold significant value of seed set threshold
    pt = pd.DataFrame(pd.Series(pt_dict)).transpose()
    # Number of qualified. If 50, it likely didn't end bistability within the ranges of the perturbation
    nq = pd.DataFrame(pd.Series(nqual_dict)).transpose()

    fc.to_csv("fc.csv", mode="a", index=False, header=False)
    pt.to_csv("pt.csv", mode="a", index=False, header=False)
    nq.to_csv("nq.csv", mode="a", index=False, header=False)

    return fc, pt, nq


if __name__ == "__main__":
    seed_set_ths = tuple(pd.read_csv("./seed_sets.csv")["off_thresh"])
    Parallel(n_jobs=-1)(delayed(dr_analysis)(dr_num, seed_set_ths) for dr_num in range(len(os.listdir("./depthRuns/")))
