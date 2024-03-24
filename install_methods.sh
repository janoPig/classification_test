#!/bin/bash

# create enviroment
python3 -m venv test_env
source test_env/bin/activate
pip install -r test_env_requirements.txt

# install methods
./install_hroch.sh
./install_rils-rols.sh
