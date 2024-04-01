import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted

class GenericPseudoClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator, t : float = 1.0, verbose : int = 0):
        self.estimator = estimator
        self.t = t
        self.verbose = verbose
    
    def fit(self, X, y):
        # transform classes to {0, 1}
        #X, y = self._validate_data(X, y, accept_sparse=False, y_numeric=False, multi_output=False)
        check_classification_targets(y)
        enc = LabelEncoder()
        y_ind = enc.fit_transform(y)
        self.classes_ = enc.classes_
        self.n_classes_ = len(self.classes_)
        if self.n_classes_ != 2:
            raise ValueError(
                "This solver needs samples of 2 classes"
                " in the data, but the data contains"
                " %r classes"
                % self.n_classes_
            )
            
        # transform data -t for negative class, t for positive class
        y_ = (y_ind-0.5)*2.0
        y_transformed = y_*self.t
        self.estimator.fit(X, y_transformed)
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        check_is_fitted(self)
        #X = self._validate_data(X, accept_sparse=False, reset=False)
        preds = self.estimator.predict(X)
        preds = np.nan_to_num(preds,nan=0)
        return self.classes_[(preds > 0.0).astype(int)]
    
    def predict_proba(self, X):
        check_is_fitted(self)
        #X = self._validate_data(X, accept_sparse=False, reset=False)
        
        preds = self.estimator.predict(X)
        preds = np.nan_to_num(preds,nan=0)
        preds = np.clip(preds, -20, 20)
        proba = 1.0/(1.0+np.exp(-preds))
        proba = np.clip(proba, 3e-7, 1.0-3e-7)
        return np.vstack([1 - proba, proba]).T

    def _more_tags(self):
        return {'binary_only': True}
