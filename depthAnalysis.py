import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import csv

# Note: columns are in a fixed order according to param_names

param_names = ['k_E', 'k_M', 'k_CD', 'k_CDS', 'k_R', 'k_RE', 'k_b', 'k_CE', 'k_I', 'k_P1', 'k_P2', 'k_DP', 'd_M', 'd_E', 'd_CD', 'd_CE', 'd_R', 'd_RP', 'd_RE', 'd_I', 'K_S', 'K_M', 'K_E', 'K_CD', 'K_CE', 'K_RP', 'K_P1', 'K_P2']

# Construct list of all names and behaviors.
all_names = []
for name in param_names:
    all_names.append(name + "_dec")
    all_names.append(name + "_inc")

scalars = np.logspace(np.log10(0.1), np.log10(10), num=100)

n = 100


# curve_dists: Column of spaces between curves
# thresholds: activation thresholds
# show_plot: visualization of spaces

# Returns a set of 2 lists. First list is whether or not the net change is increase or decrease
# for increasing and decreasing values of parameters. Second list is the proportion of parameter
# values that the relative distance between on and off curves are getting smaller.

def inc_dec(curve_dists, thresholds, name, show_plot=False):
    # Calculate the relative distance between curves given thresholds.
    # threshold = 1 if calculating just distance between thresholds
    relative_dists = list(np.divide(curve_dists, thresholds))

    # Split up lower and upper indices into separate lists. Represents scalar multiples > 1 and < 1
    lower = relative_dists[:50]
    upper = relative_dists[50:]

    # Remove nan from lists
    clower = [x for x in lower if np.isnan(x) == False]
    cupper = [x for x in upper if np.isnan(x) == False]

    # Lower 50 calculations
    # If clower has no values (reciprocated)
    if len(clower) < 2:
        lcounter = 0
        lower_inc = "."
        lprop = -1
    else:
        # Find how many index values are greater than their next index value (reciprocated)
        lcounter = 0
        for i in range(len(clower) - 1):
            if clower[i + 1] < clower[i]:
                lcounter += 1
        lprop = lcounter / len(clower)

        # Determine overall behavior from last index to first (reciprocated)
        lower_inc = "+"
        if clower[-1] > clower[0]:
            lower_inc = "-"

    # Upper 50 calculations
    if len(cupper) < 2:
        ucounter = 0
        upper_inc = "."
        uprop = -1
    else:
        ucounter = 0
        for i in range(len(cupper) - 1):
            if cupper[i + 1] >= cupper[i]:
                ucounter += 1
        uprop = ucounter / len(cupper)

        upper_inc = "+"
        if cupper[-1] < cupper[0]:
            upper_inc = "-"

    if show_plot:
        plt.figure()
        plt.title(name)
        plt.plot(relative_dists)
        # plt.savefig(f'relDists/{param}.png')

    return ([lower_inc, upper_inc], [lprop, uprop])


# Assigns overall behavior to parameter and direction

def assigner(data, param_names):
    # First list of data_dict is distance between curves.
    # Second list of data_dict is E2F activation threshold.
    data_dict = {}
    for param in param_names:
        subset = data[data["an_type"] == param].sort_values(param)
        rel = subset[[param, "dOnOff", "on_th"]]
        data_dict[param] = [inc_dec(rel['dOnOff'], rel['on_th'], param)[0]]
        data_dict[param].append(inc_dec(rel['on_th'], 1, param)[0])


    # Shallower, deeper, and dne behaviors
    deeper = []
    shallower = []
    narrower = []
    dne = []

    for key in data_dict.keys():
        # Shallower or deeper
        if data_dict[key][0][0] == "+" and data_dict[key][1][0] == "+":
            deeper.append(key + "_dec")
        if data_dict[key][0][1] == "+" and data_dict[key][1][1] == "+":
            deeper.append(key + "_inc")
        if data_dict[key][0][0] == "+" and data_dict[key][1][0] == "-":
            shallower.append(key + "_dec")
        if data_dict[key][0][1] == "+" and data_dict[key][1][1] == "-":
            shallower.append(key + "_inc")

        # for DNE
        if data_dict[key][0][0] == "." or data_dict[key][1][0] == ".":
            dne.append(key + "_dec")
        if data_dict[key][0][1] == "." or data_dict[key][1][1] == ".":
            dne.append(key + "_inc")

        if data_dict[key][0][0] == "-" and data_dict[key][1][0] == "+":
            narrower.append(key + "_dec")
        if data_dict[key][0][1] == "-" and data_dict[key][1][1] == "+":
            narrower.append(key + "_inc")
        if data_dict[key][0][0] == "-" and data_dict[key][1][0] == "-":
            narrower.append(key + "_dec")
        if data_dict[key][0][1] == "-" and data_dict[key][1][1] == "-":
            narrower.append(key + "_inc")


    return (deeper, shallower, narrower, dne)


# Check if files are incomplete.
incomp_list = []
for i in range(n):
    with open(f"depthRuns/DR{i}.csv", "r") as file:
        if len(file.readlines()) != 2801:
            incomp_list.append(i)

for i in range(n):
    # If a file was incomplete, skip and log it. Otherwise continue.
    if i in incomp_list:
        with open("runs.log", "a") as log:
            log.write(f"Index {i} incomplete. Analysis skipped.\n")
        continue
    else:
        data = pd.read_csv(f"depthRuns/DR{i}.csv")

    deeper, shallower, narrower, dne = assigner(data, param_names)

    row_dict = {}
    for name in all_names:
        if name in shallower:
            row_dict[name] = "s"
        elif name in deeper:
            row_dict[name] = "d"
        elif name in dne:
            row_dict[name] = "x"
        else:
            row_dict[name] = "n"

    row_list = []
    for key in row_dict.keys():
        row_list.append(row_dict[key])

    # Write the behaviors of parameters
    with open("paramBehaviors.csv", 'a+', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(row_list)

    row_dict = {}
    for param in param_names:
        subset = data[data["an_type"] == param].sort_values(param)
        rel = subset[[param, "dOnOff", "on_th"]]
        props = inc_dec(rel['dOnOff'], rel['on_th'], param)[1]
        row_dict[param + "_dec"] = props[0]
        row_dict[param + "_inc"] = props[1]

    row_list = []
    for key in row_dict.keys():
        row_list.append(row_dict[key])

    # Write distance proportions
    with open("distProps.csv", 'a+', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(row_list)
