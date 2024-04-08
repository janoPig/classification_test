import numpy as np, pandas as pd, sympy as sp
from HROCH import NonlinearLogisticRegressor
from rils_rols.rils_rols import RILSROLSBinaryClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from methods_generators import *
from pmlb import fetch_data, classification_dataset_names
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score, make_scorer
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from optuna import create_study
import os, sys, time, argparse
from joblib import Parallel, delayed

from sklearn.metrics import log_loss
import optuna
from pmlb_datasets import DATASETS

DEBUG = True
TEST_SIZE = 0.3
OPTUNA_TRIALS = 3 if DEBUG else 100
SEEDS = [57302, 92067, 33585, 41729, 66580]

# (needs_proba, metric)
evaluated_metrics = [
    (True, log_loss),
    (True, roc_auc_score),
    (False, accuracy_score),
]

# method_name: (method_class, method_generator)
tested_methods = {
    'RILS-ROLS': (RILSROLSBinaryClassifier, rils_rols_generator),
    'HROCH': (NonlinearLogisticRegressor, hroch_generator),
    'CatBoost': (CatBoostClassifier, cb_generator),
    'GradientBoosting': (GradientBoostingClassifier, gb_generator),
    'LGBM': (LGBMClassifier, lgbm_generator),
    'XGB': (XGBClassifier, xgb_generator),
    'DecisionTree': (DecisionTreeClassifier, dt_generator),
    'LogisticRegression': (LogisticRegression, lr_generator),
    'KNeighbors': (KNeighborsClassifier, kn_generator),
    'RandomForest': (RandomForestClassifier, rf_generator),
    'SVC': (SVC, svc_generator),
}

def get_model_string(est):
    if isinstance(est, NonlinearLogisticRegressor):
        return str(sp.parse_expr(est.sexpr_))
    elif isinstance(est, RILSROLSBinaryClassifier):
        return est.model_string()
    return ""

def objective(trial, method_generator, X, y, random_state):
    
    clf = method_generator(trial, random_state)

    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    log_loss_scorer = make_scorer(log_loss, response_method="predict_proba", greater_is_better=False)
    scores = -cross_val_score(clf, X, y, scoring=log_loss_scorer, cv=kf)
    
    return np.max([np.mean(scores), np.median([scores])])

def tune_method(method_name, method_generator, X, y, random_state):
    
    study = create_study(study_name=method_name, direction='minimize')
    study.optimize(lambda trial: objective(trial, method_generator, X, y, random_state), n_trials=OPTUNA_TRIALS)
    
    return study.best_value, study.best_params

def parse_arguments():
    parser = argparse.ArgumentParser(description='Fit classification methon on ')
    parser.add_argument('--method_name', type=str, help='Name of the method')
    parser.add_argument('--dataset_name', type=str, help='Name of pmlb dataset')
    parser.add_argument('--random_seed', type=int, help='Random seed index')
    args = parser.parse_args()
    return args.method_name, args.dataset_name, args.random_seed

if __name__ == '__main__':
    method_name, dataset_name, random_seed = parse_arguments()
    if not method_name in tested_methods:
        print(f'invalid method name: {method_name}')
        sys.exit(1)
    if not dataset_name in DATASETS:
        print(f'invalid dataset name: {dataset_name}')
        sys.exit(1)
    if random_seed < 0 or random_seed >= len(SEEDS):
        print(f'invalid random seed index: {random_seed}')
    
    X, y = fetch_data(dataset_name, return_X_y=True)
    random_state = SEEDS[random_seed]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=random_state)
    clf_class, clf_generator = tested_methods[method_name]
    
    optuna.logging.set_verbosity(optuna.logging.CRITICAL)
    train_score, best_params = tune_method(method_name, clf_generator, X_train, y_train, random_state)
    print(f'Training {method_name} on {dataset_name} score: {train_score}, best params: {best_params}')
    est = clf_class(**best_params)
    
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
    df_results = pd.DataFrame(columns=['dataset', 'method', 'time', 'model_string', 'random_seed', *metric_train, *metric_test])
    df_results.loc[len(df_results.index)] = [dataset_name, method_name, elapsed, model_string, random_state, *score_train, *score_test]
    df_results.to_csv(f'results/results_{method_name}_{dataset_name}_{random_seed}.csv', index=False)
    
    