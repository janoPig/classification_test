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
from optuna import Trial

# tunned hyperparameters for each model
def rils_rols_generator(trial : Trial):
    params = {
        'max_fit_calls': trial.suggest_int('iterations', 1000, 100000, log=True),
    }
    return RILSROLSBinaryClassifier(**params)

def hroch_generator(trial : Trial):
    params = {
        'iter_limit': trial.suggest_int('iter_limit', 1000, 100000, log=True),
    }
    return NonlinearLogisticRegressor(**params)

def cb_generator(trial : Trial):
    params = {
        'silent': True,
        'objective':'Logloss',
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'depth': trial.suggest_int('depth', 1, 12),
        'l2_leaf_reg' : trial.suggest_float('learning_rate', 0.1, 100, log=True),
        'random_strength' : trial.suggest_float('random_strength', 0.1, 2),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.01, 0.5),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 50),
        'boosting_type': trial.suggest_categorical('boosting_type', ['Ordered', 'Plain']),
        'eval_metric' : 'Logloss',
    }

    return CatBoostClassifier(**params)

def gb_generator(trial : Trial):
    params = {
        'loss': trial.suggest_categorical(name='loss', choices=['deviance', 'exponential']),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-2, 1),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 200),
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        'max_leaf_nodes': None,
        'n_iter_no_change': trial.suggest_int('n_iter_no_change', 1, 20),
        'validation_fraction': trial.suggest_discrete_uniform('validation_fraction', 0.01, 0.31, 0.01)
    }
    return GradientBoostingClassifier(**params)

def lgbm_generator(trial : Trial):
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': trial.suggest_categorical(name='boosting_type', choices=['gbdt', 'dart', 'goss']),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        'n_estimators': trial.suggest_int('n_estimators', 10, 100),  # 200-6000 by 200
        'deterministic': True,
        'force_row_wise': True,
        'n_jobs': 1,
    }
    if 2 ** params['max_depth'] > params['num_leaves']:
        params['num_leaves'] = 2 ** params['max_depth']
    return LGBMClassifier(**params)

def xgb_generator(trial : Trial):
    params = {
        'booster': trial.suggest_categorical(name='booster', choices=['gbtree', 'dart']),
        'n_estimators': trial.suggest_int('n_estimators', 10, 100),
        'objective': 'binary:logistic',
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-5, 1e2),
        'alpha': trial.suggest_loguniform('alpha', 1e-5, 1e2),
        'gamma': trial.suggest_discrete_uniform('gamma', 0, 0.5, 0.1),
        'eta': trial.suggest_loguniform('eta', 1e-8, 1),
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        'eval_metric': 'logloss',
        'tree_method': 'exact',
        'nthread': 1,
        'use_label_encoder': False,
    }
    return XGBClassifier(**params)

def dt_generator(trial : Trial):
    params = {
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'min_weight_fraction_leaf': 0.0,
        'max_features': trial.suggest_categorical('max_features', [None, 'sqrt', 'log2']),
        'max_leaf_nodes': None,
    }
    return DecisionTreeClassifier(**params)

def lr_generator(trial : Trial):
    params = {}
    params['solver'] = trial.suggest_categorical(name='solver',
                                                 choices=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
    params['dual'] = False
    params['penalty'] = 'l2'
    params['C'] = trial.suggest_loguniform('C', 1e-4, 1e4)
    params['l1_ratio'] = None
    if params['solver'] == 'liblinear':
        params['penalty'] = trial.suggest_categorical(name='penalty', choices=['l1', 'l2'])
        if params['penalty'] == 'l2':
            params['dual'] = trial.suggest_categorical(name='dual', choices=[True, False])
        else:
            params['penalty'] = 'l1'

    param_grid = {'solver': params['solver'],
                  'penalty': params['penalty'],
                  'dual': params['dual'],
                  'multi_class': 'auto',
                  'l1_ratio': params['l1_ratio'],
                  'C': params['C'],
                  'max_iter':5000
                  }
    return LogisticRegression(**param_grid)

def kn_generator(trial : Trial):
    params = {
        'n_neighbors': trial.suggest_int('n_neighbors', 1, 100),
        'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
        'p': trial.suggest_int('p', 1, 5),
        'metric': trial.suggest_categorical('metric', ['euclidean', 'minkowski'])
    }
    return KNeighborsClassifier(**params)

def rf_generator(trial : Trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 10, 100),
        'criterion': trial.suggest_categorical(name='criterion', choices=['gini', 'entropy']),
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        'max_features': trial.suggest_categorical('max_features', [None, 'sqrt', 'log2']),
        'bootstrap': True,
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
    }
    return RandomForestClassifier(**params)

def svc_generator(trial : Trial):
    params = {
        'kernel': trial.suggest_categorical(name='kernel', choices=['poly', 'rbf', 'linear', 'sigmoid']),
        'C': trial.suggest_loguniform('C', 1e-2, 1e5),
        'gamma': trial.suggest_categorical(name='gamma', choices=['scale', 'auto']),
        'degree': trial.suggest_int('degree', 2, 5),
        'class_weight': trial.suggest_categorical(name='class_weight', choices=[None, 'balanced']),
        'coef0': trial.suggest_discrete_uniform('coef0', 0, 10, 0.1),
        'probability': True
    }
    return SVC(**params)
