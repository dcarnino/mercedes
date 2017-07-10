from sklearn import metrics, model_selection
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from scipy import stats
import numpy as np
import pandas as pd
from tqdm import tqdm



class correlation_ensembling(BaseEstimator, RegressorMixin):


    def __init__(self, estimator_list, keep_p=0.1, predict_median=True):

        self.estimator_list_ = estimator_list
        self.keep_p = keep_p
        self.keep_n = int(round(len(self.estimator_list_)*self.keep_p))
        self.predict_median = predict_median

    def fit(self, X, y, verbose=0):

        X, y = check_X_y(X, y)

        for ix, estimator in tqdm(enumerate(self.estimator_list_)):
            estimator.fit(X, y)

        self.is_fitted_ = True

    def predict(self, X, verbose=0):

        check_is_fitted(self, ['is_fitted_'])

        X = check_array(X)

        y_pred = np.array([estimator.predict(X) for estimator in self.estimator_list_])

        df_pred = pd.DataFrame(y_pred.T)
        Y_corr = df_pred.corr().abs().values
        y_corr = np.sum(Y_corr, axis=1)
        pred_list = [np.argmin(y_corr)]
        while len(pred_list) < self.keep_n:
            y_corr = np.sum(Y_corr[:,pred_list], axis=1)
            pred_list.append(np.argmin(y_corr))

        y_pred = y_pred[pred_list, :]

        if self.predict_median:
            y_pred = np.median(y_pred, axis=0)
        else:
            y_pred = np.mean(y_pred, axis=0)

        return y_pred
