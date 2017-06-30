"""
    Name:           define_model.py
    Created:        30/6/2017
    Description:    Global definition of models.
"""
# ===========================
# Modules
# ===========================
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.linear_model import SGDRegressor, LassoCV, LarsCV, RidgeCV, ElasticNetCV, ARDRegression, BayesianRidge
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor

# ===========================
# Layers
# ===========================
n_jobs = 28
n_est = 100
reg_first_layer = [ ( "RF0", RandomForestRegressor(n_estimators=n_est, criterion="mse", n_jobs=n_jobs) ),
                    ( "XGB0", XGBRegressor(n_estimators=n_est*2, objective='reg:logistic', gamma=0, reg_lambda=1,
                                            min_child_weight=2, learning_rate=0.05, subsample=0.5, colsample_bytree=0.6, max_depth=4, nthread=n_jobs) ),
                    ( "XGB1", XGBRegressor(n_estimators=n_est*2, objective='reg:logistic', gamma=0, reg_lambda=1,
                                            min_child_weight=4, learning_rate=0.02, subsample=0.8, colsample_bytree=0.8, max_depth=4, nthread=n_jobs) ),
                    ( "XGB2", XGBRegressor(n_estimators=n_est*2, objective='reg:logistic', gamma=0, reg_lambda=1,
                                            min_child_weight=4, learning_rate=0.05, subsample=0.5, colsample_bytree=0.6, max_depth=4, nthread=n_jobs) ),
                    ( "XGB3", XGBRegressor(n_estimators=n_est*2, objective='reg:logistic', gamma=0, reg_lambda=1,
                                            min_child_weight=2, learning_rate=0.02, subsample=0.5, colsample_bytree=0.6, max_depth=4, nthread=n_jobs) ),
                    ( "XGB4", XGBRegressor(n_estimators=n_est*2, objective='reg:logistic', gamma=0, reg_lambda=1,
                                            min_child_weight=2, learning_rate=0.08, subsample=0.5, colsample_bytree=0.6, max_depth=4, nthread=n_jobs) ),
                    ( "LGBM", lgb.LGBMRegressor(objective='regression', n_estimators=n_est*2, num_leaves=31,
                                                 subsample=0.5, colsample_bytree=0.6, max_depth=4, nthread=n_jobs) ),
                    ( "MLP0", MLPRegressor(hidden_layer_sizes=(100,100,100), activation='relu', learning_rate='adaptive',
                                            early_stopping=True, validation_fraction=0.1, alpha=0.001, solver='adam') ),
                    ( "MLP1", MLPRegressor(hidden_layer_sizes=(1000,1000), activation='relu', learning_rate='adaptive',
                                            early_stopping=True, validation_fraction=0.1, alpha=0.01, solver='adam') ),
                    ( "MLP2", MLPRegressor(hidden_layer_sizes=(50,50,50,50), activation='relu', learning_rate='adaptive',
                                            early_stopping=True, validation_fraction=0.1, alpha=1e-05, solver='adam') ),
                    #( "BaggingSVR", BaggingRegressor(SVR(C=1.0, kernel='rbf', gamma='auto', shrinking=True, tol=0.001),
                    #                                 n_estimators=n_est//2, max_samples=4./(n_est//2), bootstrap=True, n_jobs=n_jobs) ),
                    ( "kNN", KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto', n_jobs=n_jobs) ),
                    ( "ExtraTrees0",  ExtraTreesRegressor(n_estimators=n_est, criterion='mse', bootstrap=False, n_jobs=n_jobs) ),
                    ( "ExtraTrees1",  ExtraTreesRegressor(n_estimators=n_est, criterion='mse', bootstrap=True, n_jobs=n_jobs) ),
                    ( "GBR", GradientBoostingRegressor(loss='huber', learning_rate=0.02, n_estimators=n_est, subsample=0.5,
                                                        criterion='friedman_mse', min_samples_split=2, min_samples_leaf=2, max_depth=4) ),
                    ( "AdaBoostRF", AdaBoostRegressor(base_estimator=RandomForestRegressor(n_estimators=n_est//2, n_jobs=n_jobs),
                                                      n_estimators=n_est//10, learning_rate=0.9) ),
                    ( "AdaBoostExtraTrees", AdaBoostRegressor(base_estimator=ExtraTreesRegressor(n_estimators=n_est//2, bootstrap=False, n_jobs=n_jobs),
                                                              n_estimators=n_est//10, learning_rate=0.9) )
                    ]


##### Final layer classifier
reg_final_layer = XGBRegressor(n_estimators=1000, objective='reg:logistic', gamma=0, reg_lambda=1, min_child_weight=4,
                               learning_rate=0.02, subsample=0.8, colsample_bytree=0.8, max_depth=4, nthread=n_jobs)
#reg_final_layer = AdaBoostRegressor(base_estimator=ExtraTreesRegressor(n_estimators=1000, bootstrap=False, n_jobs=n_jobs),
#                                    n_estimators=5, learning_rate=0.8)
#reg_final_layer = ExtraTreesRegressor(n_estimators=1000, bootstrap=False, n_jobs=n_jobs)
