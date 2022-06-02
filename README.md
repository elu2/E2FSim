
# E2F Simulation

## About
Simulation and analysis of the E2F network described and mathematically defined in the below 2008 paper. Of primary focus are the initial conditions where E2F is turned on ($EE_{On}$), and turned off ($EE_{Off}$). Initial steps include generating random parameter sets that meet 4 criteria:
1. (Weak) Switch-like behavior: $\Delta EE_{Off} = max(EE_{Off}) -min(EE_{Off}) > 0.1$
2. Bistability: Bistable region is > 0.2
3. Resettability: $\lim\limits_{[S] \to 0}\Delta EE =0$ for $\Delta EE=|EE_{Off} - EE_{On}|$
4. Biologically sound: $0.5\leq [S]_{Off}\leq10$

Then, systematically perturbing each parameter of each seed set to simulate and analyze.

## Scripts

1. **PreSeedInit.py**: Generates randomized parameter sets within 0.1x and 10x the original parameter values listed in the 2008 paper. Produces `pre_seed_sets.csv` (Multipliers sampled from logspace)

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

# References
Yao, G., Lee, T., Mori, S. et al. A bistable Rb–E2F switch underlies the restriction point. Nat Cell Biol 10, 476–482 (2008). https://doi.org/10.1038/ncb1711
