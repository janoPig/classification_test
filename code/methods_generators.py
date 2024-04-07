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
def rils_rols_generator(trial : Trial, random_state):
    params = {
        "random_state": random_state,
        "max_fit_calls": trial.suggest_int("iterations", 1000, 10000000, log=True),
    }
    return RILSROLSBinaryClassifier(**params)

def hroch_generator(trial : Trial, random_state):
    params = {
        "random_state": random_state,
        "iter_limit": trial.suggest_int("iter_limit", 1000, 1000000, log=True),
    }
    return NonlinearLogisticRegressor(**params)

def cb_generator(trial : Trial, random_state):
    params = {
        "random_state": random_state,
        "silent": True,
        "iterations": trial.suggest_int("iterations", 500, 5000, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "depth": trial.suggest_int("depth", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
    }
    return CatBoostClassifier(**params)

def gb_generator(trial : Trial, random_state):
    params = {
        "random_state": random_state,
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
    }
    return GradientBoostingClassifier(**params)

def lgbm_generator(trial : Trial, random_state):
    params = {
        "random_state": random_state,
        "verbose": -1,
        "iterations": trial.suggest_int("iterations", 500, 5000, log=True),
    }
    return LGBMClassifier(**params)

def xgb_generator(trial : Trial, random_state):
    params = {
        "random_state": random_state,
        "iterations": trial.suggest_int("iterations", 500, 5000, log=True),
    }
    return XGBClassifier(**params)

def dt_generator(trial : Trial, random_state):
    params = {
        "random_state": random_state,
        "criterion": trial.suggest_categorical('criterion', ['gini', 'entropy']),
        "max_depth":trial.suggest_int('max_depth', 2, 32),
        "min_samples_split": trial.suggest_int('min_samples_split', 2, 20),
        "min_samples_leaf": trial.suggest_int('min_samples_leaf', 1, 20),
    }
    return DecisionTreeClassifier(**params)

def lr_generator(trial : Trial, random_state):
    params = {
        "C": trial.suggest_float("C", 1e-6, 1e+6, log=True),
    }
    return LogisticRegression(**params)

def kn_generator(trial : Trial, random_state):
    params = {
        "n_neighbors": trial.suggest_int("n_neighbors", 1, 50, log=True),
        "leaf_size": trial.suggest_int("leaf_size", 1, 50),
        "weights" : trial.suggest_categorical('weights', ['uniform', 'distance']),
        "algorithm" : trial.suggest_categorical('algorithm', ['ball_tree', 'kd_tree', 'brute']),
        "p": trial.suggest_float("p", 1.0, 5.0),
    }
    return KNeighborsClassifier(**params)

def rf_generator(trial : Trial, random_state):
    params = {
        "random_state": random_state,
        "criterion": trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss']),
        "max_depth":trial.suggest_int('max_depth', 2, 32),
    }
    return RandomForestClassifier(**params)

def svc_generator(trial : Trial, random_state):
    params = {
        "random_state": random_state,
        "C": trial.suggest_float("C", 1e-6, 1e+6, log=True),
        "kernel": trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"]),
        "degree": trial.suggest_int("degree", 1, 8),
    }
    return SVC(**params)