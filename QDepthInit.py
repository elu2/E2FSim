import pandas as pd
import os

colnames = ['k_E', 'k_M', 'k_CD', 'k_CDS', 'k_R', 'k_RE', 'k_b', 'K_S', 'k_CE', 'k_I', 'd_M', 'd_E', 'd_CD', 'd_CE', 'd_R', 'd_RP', 'd_RE', 'd_I', 'k_P', 'k_DP', 'K_M', 'K_E', 'K_CD', 'K_CE', 'K_RP', 'K_P', "dOnOff", "off_th", "on_th"]

filenames = ['d_CD13analysis.csv', 'd_CE14analysis.csv', 'd_E12analysis.csv', 'd_I18analysis.csv', 'd_M11analysis.csv', 'd_R15analysis.csv', 'd_RE17analysis.csv', 'd_RP16analysis.csv', 'k_b6analysis.csv', 'K_CD22analysis.csv', 'k_CD2analysis.csv', 'k_CDS3analysis.csv', 'K_CE23analysis.csv', 'k_CE7analysis.csv', 'k_DP10analysis.csv', 'k_E0analysis.csv', 'K_E21analysis.csv', 'k_I8analysis.csv', 'k_M1analysis.csv', 'K_M20analysis.csv', 'K_P25analysis.csv', 'k_P9analysis.csv', 'k_R4analysis.csv', 'k_RE5analysis.csv', 'K_RP24analysis.csv', 'K_S19analysis.csv']

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


for name in filenames:
    checker("./depthLib/" + name)
