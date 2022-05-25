# Calculate fold change and maximum derivative for each parameter in both directions
# given a DRxxx.csv file.

import pandas as pd
import numpy as np

# Iterable for depth run files.
dr_range = range(5001)


# n-th order interpolation to take derivative at each point
def spline_derivatives(x, y, degree=5):
    # Fit polynomial
    poly = np.polyfit(x, y, degree)
    # Get polynomial's derivative
    ddx_poly = np.polyder(poly)
    # Apply derivative to each point
    p = np.poly1d(ddx_poly)

    return p(x)


def DR_FoldChange_MaxDer(dr_index):
    dr_df = pd.read_csv(f"./depthRuns/DR{dr_index}.csv")
    # Simplify by making new column for all 3 criteria being met
    dr_df["valid"] = dr_df["switch"] * dr_df["resettable"] * dr_df["switch"]

    # Caution of hard-coded indices 
    param_names = list(dr_df.columns[:-8])

    # Get seed set's off threshold
    sss = pd.read_csv("seed_sets.csv")
    base_off = float(sss.iloc[int(dr_index)]["off_thresh"])

    # Rows for record keeping of fold changes and maximum derivatives
    fc_dict = {"Index": dr_index}
    max_ddx_dict = {"Index": dr_index}

    # Loop through each parameter and calculate fc and md
    for i in range(len(param_names)):
        spec_df = dr_df[dr_df["an_type"] == param_names[i]]
        spec_df = spec_df[[param_names[i], "off_thresh", "valid"]]
        spec_df = spec_df.sort_values(param_names[i]).reset_index(drop=True)

        # Split parameter dataframe into increasing and decreasing halves
        lower_spec = spec_df[:50]
        # Retain only rows with all 3 criteria
        lower_spec = lower_spec[lower_spec["valid"] == True].dropna()
        # Prep data for interpolation
        lower_x = np.log10(lower_spec[param_names[i]])
        lower_traj = list(lower_spec["off_thresh"])
        # Handle when seed set was on the cusp of failing the 3 criteria
        if len(lower_traj) == 0:
            lower_fc = 0
            lower_ddx_max = 0
        else:
            # Calculate fold change
            lower_fc = (lower_traj[0] - base_off) / base_off
            # Calculate derivatives and get maximum value
            ddx_lower = spline_derivatives(lower_x, lower_traj)
            lower_ddx_max = max(abs(ddx_lower))

        upper_spec = spec_df[50:]
        upper_spec = upper_spec[upper_spec["valid"] == True].dropna()
        upper_x = np.log10(upper_spec[param_names[i]])
        upper_traj = list(upper_spec["off_thresh"])
        if len(upper_traj) == 0:
            upper_fc = 0
            upper_ddx_max = 0
        else:
            upper_fc = (upper_traj[-1] - base_off) / base_off
            ddx_upper = spline_derivatives(upper_x, upper_traj)
            upper_ddx_max = max(abs(ddx_upper))

        # Write out data to decreasing and increasing, respectively
        fc_dict[f"{param_names[i]}_dec"] = [lower_fc]
        fc_dict[f"{param_names[i]}_inc"] = [upper_fc]

        max_ddx_dict[f"{param_names[i]}_dec"] = [lower_ddx_max]
        max_ddx_dict[f"{param_names[i]}_inc"] = [upper_ddx_max]
    
    # Convert to a row of a dataframe for ease
    fc_row = pd.DataFrame(fc_dict)
    mddx_row = pd.DataFrame(max_ddx_dict)
    
    return fc_row, mddx_row


for i in dr_range:
    fc_row, mddx_row = DR_FoldChange_MaxDer(i)
    fc_row.to_csv("DR_FC.csv", mode='a', header=not os.path.exists("DR_FC.csv"), index=False)
    mddx_row.to_csv("DR_MD.csv", mode='a', header=not os.path.exists("DR_MD.csv"), index=False)
    
