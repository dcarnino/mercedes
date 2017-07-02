"""
    Name:           define_model.py
    Created:        30/6/2017
    Description:    Global definition of models.
"""
# ===========================
# Modules
# ===========================
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.linear_model import SGDRegressor, LassoCV, LarsCV, RidgeCV, ElasticNetCV, ARDRegression, BayesianRidge
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, RMSprop, Nadam, Adagrad, Adadelta, Adam, Adamax
from keras_regressor import KerasRegressor

# ===========================
# Layers
# ===========================
n_jobs = 28
n_est = 224

reg_first_layer = []

### gbr
reg_first_layer.append( ( "GBR", GradientBoostingRegressor(loss='huber', learning_rate=0.02, n_estimators=n_est, subsample=0.5,
                                                           criterion='friedman_mse', min_samples_split=2, min_samples_leaf=2, max_depth=4) ) )

### mlp
# function for model
def create_model(k_n_layers=1, k_n_units=64, k_dropout=0.5,
                 k_optimizer='rmsprop', k_init='glorot_uniform',
                 k_loss='mean_squared_error'):
    # create model
    model = Sequential()
    model.add(Dense(k_n_units, activation='relu', kernel_initializer=k_init, input_dim=551))
    model.add(Dropout(k_dropout))
    for nl in range(k_n_layers):
        model.add(Dense(k_n_units//(nl+1)+1, activation='relu', kernel_initializer=k_init))
        if k_n_units//(nl+1)+1 > 16:
            model.add(Dropout(k_dropout))
    model.add(Dense(1, activation='linear', kernel_initializer=k_init))
    # Compile model
    model.compile(loss=k_loss, optimizer=k_optimizer)
    return model
# test zipped combinations
sgd = SGD(lr=0.01, decay=5e-2, momentum=0.9, nesterov=True)
rms = RMSprop(lr=0.03, rho=0.9, epsilon=1e-08, decay=1e-3)
ada = Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
nad = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
add = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
adm = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
adx = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
glo = 'glorot_uniform'
he = 'he_normal'
k_n_layers_list = np.array((0, 1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 6, 6)) * 1
k_n_units_list = np.array((2048, 1024, 1024, 1024, 1024, 512, 256, 256, 256, 128, 128, 128, 128, 128, 256)) // 2
k_dropout_list = (0.2, 0.5, 0.3, 0.2, 0.1, 0.5, 0.5, 0.3, 0.1, 0.7, 0.5, 0.3, 0.1, 0.3, 0.5)
k_optimizer_list = (add, add, add, add, add, add, add, add, add, add, add, add, add, add, add)
k_init_list = (glo, glo, glo, glo, glo, glo, glo, glo, glo, glo, glo, glo, glo, glo, glo)
# loop
for ix, (k_n_layers, k_n_units, k_dropout, k_optimizer, k_init) \
in enumerate(zip(k_n_layers_list, k_n_units_list, k_dropout_list, k_optimizer_list, k_init_list)):
    reg_first_layer.append( ( "MLP%d"%ix, KerasRegressor(build_fn=create_model, epochs=10000, batch_size=101,
                                                         k_n_layers=k_n_layers, k_n_units=k_n_units,
                                                         k_dropout=k_dropout, k_optimizer=k_optimizer,
                                                         k_init=k_init, verbose=0) ) )

### svr
kernel_list = ('linear', 'poly', 'rbf', 'sigmoid')
for ix, kernel in enumerate(kernel_list):
    reg_first_layer.append( ( "SVR%d"%ix, SVR(C=1.0, kernel=kernel, gamma='auto', shrinking=True, tol=0.001) ) )
    reg_first_layer.append( ( "BaggingSVR%d"%ix, BaggingRegressor(SVR(C=1.0, kernel=kernel, gamma='auto', shrinking=True, tol=0.001),
                                     n_estimators=n_est//4, max_samples=4./(n_est//4), bootstrap=True, n_jobs=n_jobs) ) )

### knn
# test all combinations
n_neighbors_list = (1, 3, 5, 9)
weights_list = ('uniform', 'distance')
p_list = (1, 2, 3)
ix = 0
for n_neighbors in n_neighbors_list:
    for weights in weights_list:
        for p in p_list:
            reg_first_layer.append( ( "kNN%d"%ix, KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, p=p,
                                                                      algorithm='auto', n_jobs=n_jobs) ) )
            ix += 1

### adaboost
reg_first_layer.append( ( "AdaBoostRF", AdaBoostRegressor(base_estimator=RandomForestRegressor(n_estimators=n_est, n_jobs=n_jobs),
                                                          n_estimators=n_est//10, learning_rate=0.9) ) )
reg_first_layer.append( ( "AdaBoostExtraTrees", AdaBoostRegressor(base_estimator=ExtraTreesRegressor(n_estimators=n_est, bootstrap=True, n_jobs=n_jobs),
                                                                  n_estimators=n_est//10, learning_rate=0.9) ) )

### random forest
# test all combinations
# test zipped combinations
max_depth_list = (None, None, None, None, None, None, None, None, 2, 4, 10, 30)
min_samples_split_list = (2, 2, 2, 2, 2, 8, 0.01, 0.1, 2, 2, 2, 2)
criterion_list = ('mse', 'mse', 'mse', 'mse', 'mse', 'mse', 'mse', 'mse', 'mse', 'mse', 'mse', 'mse')
bootstrap_list = (True, True, True, True, True, True, True, True, True, True, True)
max_features_list = ('auto', 'auto', 'auto' ,'sqrt', 'auto', 'auto', 'auto', 'auto', 'auto', 'auto', 'auto')
min_samples_leaf_list = (21, 11, 5, 1, 1, 1, 1, 1, 1, 1, 1)
for ix, (max_depth, min_samples_split, criterion, bootstrap, max_features, min_samples_leaf) \
in enumerate(zip(max_depth_list, min_samples_split_list, criterion_list, bootstrap_list, max_features_list, min_samples_leaf_list)):
    reg_first_layer.append( ( "RF%d"%ix, RandomForestRegressor(n_estimators=n_est*2, max_depth=max_depth,
                                                               min_samples_split=min_samples_split, criterion=criterion,
                                                               bootstrap=bootstrap, max_features=max_features,
                                                               min_samples_leaf=min_samples_leaf, n_jobs=n_jobs) ) )
    reg_first_layer.append( ( "ExtraTrees%d"%ix, ExtraTreesRegressor(n_estimators=n_est*2, max_depth=max_depth,
                                                                     min_samples_split=min_samples_split, criterion=criterion,
                                                                     bootstrap=bootstrap, max_features=max_features,
                                                                     min_samples_leaf=min_samples_leaf, n_jobs=n_jobs) ) )

### xgboost
# test all combinations
max_depth_list = (4, 6, 8, 10)
subsample_list = (.6, .8)
colsample_bytree_list = (.6, .8)
learning_rate_list = (0.02, 0.05, 0.08, 0.2)
min_child_weight_list = (1, 2, 4)
# test zipped combinations
ix = 0
for max_depth in max_depth_list:
    for subsample in subsample_list:
        for colsample_bytree in colsample_bytree_list:
            for learning_rate in learning_rate_list:
                for min_child_weight in min_child_weight_list:
                    reg_first_layer.append( ( "XGB%d"%ix, XGBRegressor(n_estimators=n_est, objective='reg:linear',
                                                                       gamma=0, reg_lambda=1, min_child_weight=min_child_weight,
                                                                       learning_rate=learning_rate, subsample=subsample,
                                                                       colsample_bytree=colsample_bytree, max_depth=max_depth,
                                                                       nthread=n_jobs) ) )
                    ix += 1

### lgbm
# test all combinations
max_depth_list = (5, 7, 9)
subsample_list = (.5, .7, .9)
colsample_bytree_list = (.5, .7, .9)
# test zipped combinations
ix = 0
for max_depth in max_depth_list:
    for subsample in subsample_list:
        for colsample_bytree in colsample_bytree_list:
            reg_first_layer.append( ( "LGBM%d"%ix, lgb.LGBMRegressor(objective='regression', n_estimators=n_est,
                                                                     num_leaves=31, subsample=subsample,
                                                                     colsample_bytree=colsample_bytree, max_depth=max_depth,
                                                                     nthread=n_jobs) ) )
            ix += 1


##### Final layer classifier
reg_final_layer = XGBRegressor(n_estimators=11200, objective='reg:linear', gamma=0, reg_lambda=1, min_child_weight=4,
                               learning_rate=0.02, subsample=0.8, colsample_bytree=0.8, max_depth=4, nthread=n_jobs)
#reg_final_layer = AdaBoostRegressor(base_estimator=ExtraTreesRegressor(n_estimators=1000, bootstrap=False, n_jobs=n_jobs),
#                                    n_estimators=5, learning_rate=0.8)
#reg_final_layer = ExtraTreesRegressor(n_estimators=1000, bootstrap=False, n_jobs=n_jobs)
