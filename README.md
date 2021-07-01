## README

Recommended to run in range increments of 48 if simulating all models. (768 total models)

Runs are tracked in the runs.log file.

Will have to manually update the model ranges by editing the python file.

When running parallel across many models or within 1 model, run the respective dataframe-initializing script.

### SimParamParallel vs SimModelParallel
SimParamParallel breaks up a set of parameter sets and runs them in parallel. Ideal for a single model with a large parameter set (>20,000).
Manually change the number of chunks to break the set of parameters up.

SimModelParallel runs different model types in parallel to each other. Ideal for smaller parameter sets (<20,000) and many models.
Manually change the range of models to run over. Total 768 unique models.
