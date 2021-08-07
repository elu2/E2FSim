import pandas as pd
import os

colnames = ['k_E', 'k_M', 'k_CD', 'k_CDS', 'k_R', 'k_RE', 'k_b', 'k_CE', 'k_I', 'k_P', 'k_DP', 'd_M', 'd_E', 'd_CD', 'd_CE', 'd_R', 'd_RP', 'd_RE', 'd_I', 'K_S', 'K_M', 'K_E', 'K_CD', 'K_CE', 'K_RP', 'K_P', "an_type", "dOnOff", "off_th", "on_th"]

init_df = pd.DataFrame(columns=colnames)


def checker(file_name):
    if os.path.exists(file_name):
        option = input("File exists. Overwrite? [y/n]").lower()
        if option == "y":
            init_df.to_csv(file_name, index=False)
            print("New empty file created.")
        else:
            print("Previous file retained.")

    else:
        init_df.to_csv(file_name, index=False)
    
    return


checker("depthRun.csv")
