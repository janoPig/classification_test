import pandas as pd
import itertools
import multiprocessing
import subprocess
from pmlb_datasets import DATASETS
import os, sys, time, argparse

METHODS = [
    'RILS-ROLS',
    'HROCH',
    'CatBoost',
    'GradientBoosting',
    'LGBM',
    'XGB',
    'DecisionTree',
    'LogisticRegression',
    'KNeighbors',
    'RandomForest',
    'SVC',
]

def run_script(p):
    arguments = ' '.join([f'{key} {value}' for key, value in zip(parameters.keys(), p)])
    command = f'python3 ./code/fit_method.py {arguments}'
    subprocess.run(command, shell=True)
    
def parse_arguments():
    parser = argparse.ArgumentParser(description='Fit classification methods on')
    parser.add_argument('--dataset_name', type=str, help='Name of pmlb dataset')
    parser.add_argument('--method', type=str, help='Tested method or all')
    args = parser.parse_args()
    return args.dataset_name, args.method

if __name__ == '__main__':
    dataset_name, method = parse_arguments()
    if not dataset_name in DATASETS:
        print(f'invalid dataset name: {dataset_name}')
        sys.exit(1)

    if method != 'all' and not method in METHODS:
        print(f'invalid method: {method}')
        sys.exit(1)
    
    num_cpus = multiprocessing.cpu_count() - 1

    print(os.getcwd())
    print(f'Avaliable cpu count: {os.cpu_count()}')

    parameters = {
        '--dataset_name': [dataset_name],
        '--method': METHODS if method == 'all' else [method],
        '--random_seed': [0, 1, 2, 3, 4], # index to SEEDS in git method.py
    }
    parameter_combinations = list(itertools.product(*parameters.values()))

    with multiprocessing.Pool(processes=num_cpus) as pool:
        pool.map(run_script, parameter_combinations)

    directory = 'results'

    dataframes = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath)
            dataframes.append(df)
            os.remove(filepath)

    result = pd.concat(dataframes, ignore_index=True)
    result.to_csv(f'results/results_{dataset_name}.csv', index=False)
