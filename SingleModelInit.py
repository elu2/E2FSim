#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os

# Update this for name
model = [1, 1, 1, 1, 1, 1, 1, 2, 1]

model_name = ""
for state in model:
    model_name += str(state)

col_names = ["K_1", "K_2", "K_3", "K_4", "K_5", "K_6", "K_7", "K_8", "K_9", "K_10", "n_1", "n_2", "n_3", "n_4", "n_5", "n_6", "n_7", "n_8", "n_9", "n_10", "tau_MD", "tau_RP", "tau_EE", "beta_MD", "beta_RP", "beta_EE", "bistable", "rebi"]

if os.path.exists(f"SM{model_name}.csv"):
    option = input("File exists. Overwrite? [y/n]").lower()
    if option == "y":
        pd.DataFrame(columns=col_names).to_csv(f"SM{model_name}.csv", index=False)
        print(f"New empty file of model {model} created.")
    else:
        print("Previous file retained.")

else:
    pd.DataFrame(columns=col_names).to_csv(f"SM{model_name}.csv", index=False)
    print(f"New empty file of model {model} created.")
