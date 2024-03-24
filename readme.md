# Classification test

This repository contains the code for the testing symbolic classificators from HROCH and RILS-ROLS packages.

TODO: Sys req. Ubuntu 22.04, AVX2 capable CPU

## Requirements

- python 3.10
- clang-18

Can use this script [install_requirements.sh](install_requirements.sh)

## Setup enviroment and install methods HROCH and RILS-ROLS

Run install_methods.sh to create test_env python enviroment and build HROCH and RILS-ROLS packages from source code.

```bash
    ./install_methods.sh
```

HROCH binary is builded from stable branch [sr_core/classification_test](https://github.com/janoPig/sr_core/tree/classification_test) with clang++18 and python package from[HROCH/classification_test](https://github.com/janoPig/HROCH/tree/classification_test) This corresponds to version [1.4.10](https://github.com/janoPig/HROCH/releases/tag/v1.4.10) ([pypi](https://pypi.org/project/HROCH/1.4.10/))

## Experiments

- DIGEN benchmark [code/digen.ipynb](code/digen.ipynb)
- PMLB instances [code/pmlb.py](code/pmlb.py) TODO: implement this
- TODO: symbolic classification explaitable model

TODO: Add QLatice, gplearn or anothre symbolic methods to comparsion

## Evaluate results

TODO: implement this
