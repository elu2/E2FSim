from scipy.stats import weibull_min as frechet
from scipy.stats import loguniform as lu
import matplotlib.pyplot as plt
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


def rfrechet_gen(req_n, param_name, xmin=0.01, xmax=1):
    if param_name == "K_10":
        rfrechet = []
        while len(rfrechet) < req_n:
            gen = frechet.rvs(1.6, loc=-1, scale=1, size=req_n)
            mask = (gen > xmin) * (gen < xmax)
            in_range = list(gen[mask])
            rfrechet.extend(in_range)
        
        rfrechet = rfrechet[:req_n]

        return rfrechet
    
    if param_name == "K_8":
        rfrechet = []
        while len(rfrechet) < req_n:
            gen = frechet.rvs(.8, loc=-1, scale=1, size=req_n)
            mask = (gen > xmin) * (gen < xmax)
            in_range = list(gen[mask])
            rfrechet.extend(in_range)
        
        rfrechet = rfrechet[:req_n]

    if param_name == "n_8":
        rfrechet = []
        while len(rfrechet) < req_n:
            gen = frechet.rvs(.6, loc=.9, scale=2, size=req_n)
            mask = (gen > xmin) * (gen < xmax)
            in_range = list(gen[mask])
            rfrechet.extend(in_range)
        
        rfrechet = rfrechet[:req_n]
    
    if param_name == "beta_MD":
        rfrechet = []
        while len(rfrechet) < req_n:
            gen = frechet.rvs(.4, loc=0, scale=.25, size=req_n)
            mask = (gen > xmin) * (gen < xmax)
            in_range = list(gen[mask])
            rfrechet.extend(in_range)
        
        rfrechet = rfrechet[:req_n]
        
        return rfrechet
    
    
for param in K_class:
    if param == "K_10":
        params[param] = rfrechet_gen(n_params, param, xmin=K_range[0], xmax=K_range[1])
    elif param == "K_8":
        params[param] = rfrechet_gen(n_params, param, xmin=K_range[0], xmax=K_range[1])
    elif param == "K_9":
        params[param] = list(lu.rvs(K_range[0], K_range[1], size=n_params))
    else:
        params[param] = list(lu.rvs(.01, 0.155, size=n_params))
    
for param in n_class:
    if param == "n_8":
        params[param] = rfrechet_gen(n_params, param, xmin=n_range[0], xmax=n_range[1])
    else:
        params[param] = list(lu.rvs(n_range[0], n_range[1], size=n_params))
    
for param in tau_class:
    params[param] = list(lu.rvs(tau_range[0], tau_range[1], size=n_params))
    
for param in beta_class:
    if param == "beta_MD":
        params[param] = rfrechet_gen(n_params, param, xmin=beta_range[0], xmax=beta_range[1])
    else:
        params[param] = list(lu.rvs(beta_range[0], beta_range[1], size=n_params))

pd.DataFrame(params).to_csv("parameters.csv", index=False)
        