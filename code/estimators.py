import sympy as sp
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
from est_generators import *

# est_name: (est_class, est_generator)
ESTIMATORS = {
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
    return ''
