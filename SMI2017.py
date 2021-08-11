import pandas as pd
import os

col_names = ['k_E', 'k_M', 'k_CD', 'k_CDS', 'k_R', 'k_RE', 'k_b', 'k_CE', 'k_I', 'k_P1', 'k_P2', 'k_DP', 'd_M', 'd_E', 'd_CD', 'd_CE', 'd_R', 'd_RP', 'd_RE', 'd_I', 'K_S', 'K_M', 'K_E', 'K_CD', 'K_CE', 'K_RP', 'K_P1', 'K_P2', "bistable", "rebi"]

if os.path.exists(f"SM{model_name}.csv"):
    option = input("File exists. Overwrite? [y/n]").lower()
    if option == "y":
        pd.DataFrame(columns=col_names).to_csv(f"SM{model_name}.csv", index=False)
        print(f"New empty file of model {model} created.")
    else:
        print("Previous file retained.")

else:
    pd.DataFrame(columns=col_names).to_csv(f"results2017.csv", index=False)
    print(f"New empty file of model 2017 model created.")
