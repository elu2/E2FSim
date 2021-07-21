from scipy.stats import loguniform as lu
import numpy as np
import pandas as pd

n_params = 10000000

params = {
    "K_1": [],
    "K_2": [],
    "K_3": [],
    "K_4": [],
    "K_5": [],
    "K_6": [],
    "K_7": [],
    "K_8": [],
    "K_9": [],
    "K_10": [],
    "n_1": [],
    "n_2": [],
    "n_3": [],
    "n_4": [],
    "n_5": [],
    "n_6": [],
    "n_7": [],
    "n_8": [],
    "n_9": [],
    "n_10": [],
    "tau_MD": [],
    "tau_RP": [],
    "tau_EE": [],
    "beta_MD": [],
    "beta_RP": [],
    "beta_EE": []
}

K_range = [0.01, 1]
n_range = [1, 10]
tau_range = [.2, 20]
beta_range = [.1, 10]

K_class = ["K_1", "K_2", "K_3", "K_4", "K_5", "K_6", "K_7", "K_8", "K_9", "K_10"]
n_class = ["n_1", "n_2", "n_3", "n_4", "n_5", "n_6", "n_7", "n_8", "n_9", "n_10"]
tau_class = ["tau_MD", "tau_RP", "tau_EE"]
beta_class = ["beta_MD", "beta_RP", "beta_EE"]

for param in K_class:
    params[param] = list(lu.rvs(K_range[0], K_range[1], size=n_params))
    
for param in n_class:
    params[param] = list(lu.rvs(n_range[0], n_range[1], size=n_params))
    
for param in tau_class:
    params[param] = list(lu.rvs(tau_range[0], tau_range[1], size=n_params))
    
for param in beta_class:
    params[param] = list(lu.rvs(beta_range[0], beta_range[1], size=n_params))

pd.DataFrame(params).to_csv("parameters.csv", index=False)
        