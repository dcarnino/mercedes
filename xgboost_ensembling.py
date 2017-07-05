from xgboost import XGBRegressor
from sklearn import metrics, model_selection
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import numpy as np



class XGBRegressor_ensembling(BaseEstimator, RegressorMixin):


    def __init__(self, n_folds=5, early_stopping_rounds=10, eval_metric=metrics.r2_score, predict_median=False,
                 max_depth=3, learning_rate=0.1, n_estimators=100000, silent=True,
                 objective='reg:linear', nthread=None,
                 gamma=0, min_child_weight=1, max_delta_step=0, subsample=1,
                 colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1,
                 scale_pos_weight=1, base_score=0.5, seed=None, missing=None):

        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.silent = silent
        self.objective = objective
        self.nthread = nthread
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.base_score = base_score
        self.seed = seed
        self.missing = missing

        self.n_folds = n_folds
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.predict_median = predict_median


    def fit(self, X, y, sample_weight=None, verbose=0):

        X, y = check_X_y(X, y)

        self.estimator_list_ = [ XGBRegressor(max_depth=self.max_depth, learning_rate=self.learning_rate,
                                      n_estimators=self.n_estimators, silent=self.silent,
                                      objective=self.objective, nthread=self.nthread,
                                      gamma=self.gamma, min_child_weight=self.min_child_weight, max_delta_step=self.max_delta_step,
                                      subsample=self.subsample, colsample_bytree=self.colsample_bytree, colsample_bylevel=self.colsample_bytree,
                                      reg_alpha=self.reg_alpha, reg_lambda=self.reg_lambda, scale_pos_weight=self.scale_pos_weight,
                                      base_score=self.base_score, seed=self.seed, missing=self.missing) for fold in range(self.n_folds)]

        cv = model_selection.KFold(n_splits=self.n_folds, shuffle=True)

        for fold_cnt, (train_index, test_index) in enumerate(cv.split(X)):

            if verbose >= 1: print("XGB fold %d/%d..."%(fold_cnt+1,self.n_folds))

            X_train, X_test = X[valtrain_index], X[valtest_index]
            y_train, y_test = y[valtrain_index], y[valtest_index]

            self.estimator_list_[fold_cnt].fit(X_train, y_train, sample_weight=sample_weight,
                                              eval_set=[(X_test, y_test)], eval_metric=self.eval_metric,
                                              early_stopping_rounds=self.early_stopping_rounds, verbose=verbose)


    def predict(self, X):

        check_is_fitted(self, ['estimator_list_'])

        X = check_array(X)

        y_pred = [estimator.predict(X) for estimator in self.estimator_list_]
        if self.predict_median:
            y_pred = np.median(y_pred, axis=0)
        else:
            y_pred = np.mean(y_pred, axis=0)
