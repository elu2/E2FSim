## README

PreSeedInit.py: Generates randomized parameter sets within 0.1x and 10x the original parameter values listed in the 2017 paper. Produces pre_seed_sets.csv (Multipliers sampled from logspace)

SeedSim.py: Runs simluation on each parameter set from pre_seed_sets.csv. Produces pre_seed_results.csv

SeedSelect.py: Filters results from pre_seed_results.csv based on the 4 criteria. Produces seed_sets.csv.

QDIR.py: Initializes DRXXX.csv files to store simulation data. Initializes DRXXX.csv.

GDPR.py: Using the seeding parameter sets, perturbs each parameter from 0.1x to 10x as a new parameter set. 2800 total. Produces DPXXX.csv.

SDPR.py: Chunks a parameter set (DPXXX.csv) to run simulations in parallel and output reuslts to an initialized DRXXX.csv file.

Runs are tracked in the runs.log file.

Refer to E2FSimulation.pdf for which scripts to run.
