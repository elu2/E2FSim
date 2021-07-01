#!/usr/bin/env python
# coding: utf-8

from random import uniform
from math import log, exp
import pandas as pd

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

n_sim = 20000
for i in range(n_sim):
    for param in K_class:
        params[param].append(round(exp(uniform(log(K_range[0]), log(K_range[1])))), 4)

    for param in n_class:
        params[param].append(round(exp(uniform(log(n_range[0]), log(n_range[1])))), 4)

    for param in tau_class:
        params[param].append(round(exp(uniform(log(tau_range[0]), log(tau_range[1])))), 4)

    for param in beta_class:
        params[param].append(round(exp(uniform(log(beta_range[0]), log(beta_range[1])))), 4)

pd.DataFrame(params).to_csv("parameters.csv", index=False)
