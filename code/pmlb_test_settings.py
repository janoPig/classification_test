from sklearn.metrics import log_loss, roc_auc_score, accuracy_score

DEBUG = False
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