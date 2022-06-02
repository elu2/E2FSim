
## E2F Simulation

1. **PreSeedInit.py**: Generates randomized parameter sets within 0.1x and 10x the original parameter values listed in the 2017 paper. Produces `pre_seed_sets.csv` (Multipliers sampled from logspace)

2. **SeedSim.py**: Runs simluation on each parameter set from pre_seed_sets.csv. Produces `pre_seed_results.csv`

3. **SeedSelect.py**: Filters results from pre_seed_results.csv based on the 4 criteria. Produces `seed_sets.csv`.

4. **QDIR.py**: Initializes `DRXXX.csv` files to store simulation data. Initializes `DRXXX.csv`.

5. **GDPR.py**: Using the seeding parameter sets, perturbs each parameter from 0.1x to 10x as a new parameter set. 2800 total. Produces `DPXXX.csv`.

6. **SDPR.py**: Chunks a parameter set (`DPXXX.csv`) to run simulations in parallel and output results to an initialized `DRXXX.csv` file.

7. **DRAnalysis.py**: Analyze `DRXXX.csv` files for patterns associated with the perturbations. 
	* Fold change until any of 4 criteria are broken, or final tested perturbation
	* Perturbation where $EE_{Off}$ threshold reaches double or half the seed set's
	* Annotate special behaviors

# Others
* Runs are tracked in the runs.log file.

* Refer to E2FSimulation.pdf for which scripts to run. Scripts marked with HPC are resource-intensive and not recommended to run locally.
