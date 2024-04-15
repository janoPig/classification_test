# Classification test

This repository contains the code for the testing symbolic classificators from HROCH and RILS-ROLS packages.

## Requirements

- python 3.10
- clang-18
- tested on Ubuntu 22.04 compatible sytem

Can use this script [install_requirements.sh](install_requirements.sh)

## Setup enviroment and install methods HROCH and RILS-ROLS

Run install_methods.sh to create test_env python enviroment and build HROCH and RILS-ROLS packages from source code.

```bash
    ./install_methods.sh
```

HROCH binary is builded from stable branch [sr_core/classification_test](https://github.com/janoPig/sr_core/tree/classification_test) with clang++18 and python package from[HROCH/classification_test](https://github.com/janoPig/HROCH/tree/classification_test) This corresponds to version [1.4.11](https://github.com/janoPig/HROCH/releases/tag/v1.4.11) ([pypi](https://pypi.org/project/HROCH/1.4.11/))

## Experiments

- DIGEN benchmark [code/digen.ipynb](code/digen.ipynb)
- Benchmark based on PMLB datasets [code](code/evaluate.py) evaluated with [workflow](.github/workflows/pmlb.yml)

## DIGEN benchmark

[notebook](code/digen.ipynb) Because symbolic classification methods can be very successful on this type of problems, we use only benchmark.evaluate for evaluation, and omit hyperparameter optimization. 

## Benchmark based on PMLB datasets

Benchmark is implemented with using github workflow. Each estimator is tested 5 times for different dataset split. The dataset is divided into training and validation set. Classifiers are trained on the training set using optuna and cross-validation with StratifiedKFold.

- Used datasets from pmlb with conditions binary classification problem and n_rows >= 1000
- Used estimators was defined in code/estimators.py
- The parameter search space for the estimators was defined in the file code/est_generators.py. Most of the parameters were taken from the DIGEN benchmark.
    - DecisionTree search space use for max_features sqrt choose, not auto, because it is not available in the newer version.
    - LogisticRegression 'C' parameter is reduced to interval (0.1, 10) and only lbfgs solver was used, because some datasets eg. Hill_Valley have unacceptable times for running one trial.
    - Added CatBoostClassifier
    - GradientBoostingClassifier loss parameter 'log_loss'
    - Removed SVC because it is very slow for given dataset sizes, therefore optimization with Optuna will end with the first trial.
- Optuna parameters, random seeds and evaluated metrics are defined in code/pmlb_test_settings.py
    ```python
    TEST_SIZE = 0.2
    CV_SPLITS = 10
    OPTUNA_TIMEOUT = 10000
    OPTUNA_TRIALS = 5 if DEBUG else 100
    SEEDS = [57302, 92067, 33585, 41729, 66580]

    # (needs_proba, metric)
    evaluated_metrics = [
        (True, log_loss),
        (True, roc_auc_score),
        (False, accuracy_score),
    ]
    ```

### Run experiments with github workflows

Run this workflow [pmlb.yml](https://github.com/janoPig/classification_test/actions/workflows/pmlb.yml)

## Evaluate results

TODO: implement this
