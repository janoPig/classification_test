import numpy as np, pandas as pd
from estimators import ESTIMATORS, get_model_string
from pmlb_test_settings import *
from pmlb import fetch_data
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
import optuna
from pmlb_datasets import DATASETS
import sys, time, argparse, json

def objective(trial, est_generator, X, y, random_state):
    est = est_generator(trial)
    for a in ['random_state', 'seed']:
        if hasattr(est, a):
            setattr(est, a, random_state)

    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=random_state)
    scores = -cross_val_score(est, X, y, scoring='neg_log_loss', cv=cv)
    trial.set_user_attr('params', est.get_params())
    
    return np.mean(scores)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Fit classification methon on ')
    parser.add_argument('--estimator', type=str, help='Name of the method')
    parser.add_argument('--dataset_name', type=str, help='Name of pmlb dataset')
    parser.add_argument('--random_seed', type=int, help='Random seed index')
    parser.add_argument('--out_dir', type=str, help='Output dir')
    args = parser.parse_args()
    return args.estimator, args.dataset_name, args.random_seed, args.out_dir

def evaluate_estimator(est_name, dataset_name, random_seed, out_dir):
    X, y = fetch_data(dataset_name, return_X_y=True)
    random_state = SEEDS[random_seed]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=random_state)
    est_class, est_generator = ESTIMATORS[est_name]

    optuna.logging.set_verbosity(optuna.logging.INFO)
    # Make the sampler behave in a deterministic way.
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(study_name=est_name, direction='minimize', sampler=sampler)
    start = time.time()
    study.optimize(lambda trial: objective(trial, est_generator, X, y, random_state), n_trials=OPTUNA_TRIALS, timeout=OPTUNA_TIMEOUT)
    elapsed = time.time() - start
    optuna_train_score = study.best_value

    params = study.best_trial.user_attrs['params']

    print(f'Training {est_name} on {dataset_name} time:{elapsed} optuna_score: {optuna_train_score}, best params: {params}')

    est = est_class(**params)
    for a in ['random_state', 'seed']:
        if hasattr(est, a):
            setattr(est, a, random_state)
    
    start = time.time()
    est.fit(X_train, y_train)
    elapsed = time.time() - start
    preds_train, preds_test = est.predict(X_train), est.predict(X_test)
    proba_train, proba_test = est.predict_proba(X_train), est.predict_proba(X_test)

    score_train, score_test = [], []
    for need_proba, metric in evaluated_metrics:
        score_train.append(metric(y_train, proba_train[:,1]) if need_proba else metric(y_train, preds_train))
        score_test.append(metric(y_test, proba_test[:,1]) if need_proba else metric(y_test, preds_test))
    
    model_string = get_model_string(est)
    metric_train = ['train_' + m[1].__name__ for m in evaluated_metrics]
    metric_test = ['test_' + m[1].__name__ for m in evaluated_metrics]
    df_results = pd.DataFrame(columns=['dataset', 'estimator', 'time', 'model_string', 'random_seed', *metric_train, *metric_test, 'est_params'])
    df_results.loc[len(df_results.index)] = [dataset_name, est_name, elapsed, model_string, random_state, *score_train, *score_test, json.dumps(params)]
    df_results.to_csv(f'{out_dir}/results_{est_name}_{dataset_name}_{random_seed}.csv', index=False)

if __name__ == '__main__':
    est_name, dataset_name, random_seed, out_dir = parse_arguments()
    if not est_name in ESTIMATORS.keys():
        print(f'invalid estimator name: {est_name}')
        sys.exit(1)
    if not dataset_name in DATASETS:
        print(f'invalid dataset name: {dataset_name}')
        sys.exit(1)
    if random_seed < 0 or random_seed >= len(SEEDS):
        print(f'invalid random seed index: {random_seed}')
        sys.exit(1)

    evaluate_estimator(est_name, dataset_name, random_seed, out_dir )
    
    
