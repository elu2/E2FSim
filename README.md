## README

Runs are tracked in the runs.log file.

When running parallel across many models or within 1 model, run the respective dataframe-initializing script.

### SimParamParallel vs SimModelParallel vs SimDepthParallel
SimParamParallel breaks up a set of parameter sets and runs them in parallel. Ideal for a single model with a large parameter set (>20,000).
Manually change the number of chunks to break the set of parameters up.

SimModelParallel runs different model types in parallel to each other. Ideal for smaller parameter sets (<20,000) and many models.
Manually change the range of models to run over. Total 768 unique models. Suggested to run in range increments of 48.

SimDepthParallel runs different sets of parameters with 1 parameter varying in each parallel process. Reads in parameter csvs from the
required depthLib directory. Outputs are written into a single file that must be subsetted.

### Init Scripts
MultiModelInit.py -- SimModelParallel.py

SingleModelInit.py -- SimParamParallel.py

QDepthInit.py -- SimDepthParallel.py
