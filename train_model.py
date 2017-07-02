"""
    Name:           train_model.py
    Created:        29/6/2017
    Description:    Train model on mercedes data.
"""
#==============================================
#                   Modules
#==============================================
import sys
import os
import copy
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import model_selection, metrics
from scipy import sparse
from xgboost import XGBRegressor
from sklearn.decomposition import PCA, FastICA
from sklearn.feature_selection import SelectFromModel
#==============================================
#                   Files
#==============================================
import define_model


#==============================================
#                   Functions
#==============================================
def fit_stacked_regressors(X_train, y_train, n_folds=5,
                           add_raw_features=False, verbose=1):
    """
    Regressify with two layers of regressors.
    """

    ### Init cross-validation K-folds
    cv = model_selection.KFold(n_splits=n_folds, shuffle=True)

    ### Train first layer of regressors
    if verbose >= 1: print("Training 1st layer...")
    fold_cnt = 0
    X_oritrain, X2_train, y2_train = [], [], []
    for valtrain_index, valtest_index in cv.split(X_train):
        fold_cnt += 1
        if verbose >= 1: print("Processing fold number %d/%d..."%(fold_cnt,n_folds))
        # split features and target labels
        X_valtrain, X_valtest = X_train[valtrain_index], X_train[valtest_index]
        y_valtrain, y_valtest = y_train[valtrain_index], y_train[valtest_index]
        # fit classifiers
        reg_list = [copy.deepcopy(mdl) if "MLP" not in mdl[0] else mdl for mdl in define_model.reg_first_layer]
        X2_valpred = []
        for reg in reg_list:
            if verbose >= 2:
                print("%s ... "%reg[0], end='')
                sys.stdout.flush()
            if 'MLP' in reg[0]:
                reg[1].ntraintest(X_valtrain, y_valtrain)
            else:
                reg[1].fit(X_valtrain, y_valtrain)
            y_valpred = reg[1].predict(X_valtest)
            y_valpred[np.isnan(y_valpred)] = 100.
            X2_valpred.append(y_valpred.reshape((-1,1)))
            if verbose >= 3: print("(%.04f) "%(metrics.r2_score(y_valtest, y_valpred)), end='')
        if verbose >= 2: print("")
        # append to new features
        X_oritrain.append(X_valtest)
        X2_train.append(np.hstack(X2_valpred))
        y2_train.extend(y_valtest)
    X_oritrain = np.vstack(X_oritrain)
    X2_train = np.vstack(X2_train)
    y2_train = np.array(y2_train)
    # refit classifiers on all training data
    print("Refitting 1st layer on all training data...")
    reg_list = [copy.deepcopy(mdl) if "MLP" not in mdl[0] else mdl for mdl in define_model.reg_first_layer]
    for reg in reg_list:
        if verbose >= 2:
            print("%s ... "%reg[0], end='')
            sys.stdout.flush()
        if 'MLP' in reg[0]:
            reg[1].ntraintest(X_oritrain, y2_train)
        else:
            reg[1].fit(X_oritrain, y2_train)
    if verbose >= 2: print("")

    ### Init final layer
    reg_final = copy.deepcopy(define_model.reg_final_layer)

    ### Train final layer
    if verbose >= 1: print("Training 2nd layer...")
    if add_raw_features:
            reg_final.fit(np.hstack([X_oritrain[:,:183], X2_train]), y2_train)
    else:
            reg_final.fit(X2_train, y2_train)

    return reg_list, reg_final





def predict_stacked_regressors(X_test, reg_list, reg_final, add_raw_features=False, verbose=1):
    """
    Predict for two layers regressor.
    """

    ### Predict with both layers
    # layer 1
    if verbose >= 1: print("Predictions of 1st layer...")
    X2_test = []
    for reg in reg_list:
        y_subpred = reg[1].predict(X_test)
        X2_test.append(y_subpred.reshape((-1,1)))
    X2_test = np.hstack(X2_test)
    # layer 2
    if verbose >= 1: print("Predictions of 2nd layer...")
    if add_raw_features:
        X3_test = np.hstack([X_test[:,:183], X2_test])
    else:
        X3_test = X2_test
    y_pred = reg_final.predict(X3_test)

    return y_pred




def drop_correlations(X, threshold=0.99):
    """
    Get highly correlated features inplace.
    """

    # get columns to remove because correlated
    corr_indices = set([tuple(sorted([c1, c2])) for c1, c2 in zip(*np.where(X.corr().abs() > threshold)) if c1 != c2])
    corr_index = [ix_tup[1] for ix_tup in corr_indices]

    return corr_index





def main(verbose=1):
    """
    Main function.
    """

    ### Import data
    if verbose >= 1: print("Import data...")
    # train
    train_csv_name = "../data/mercedes/train.csv"
    df_train = pd.read_csv(train_csv_name)
    id_train = df_train["ID"]
    y_train = df_train["y"]
    Xb_train = df_train.iloc[:,10:]
    Xc_train = df_train.iloc[:,2:10]
    # test
    test_csv_name = "../data/mercedes/test.csv"
    df_test = pd.read_csv(test_csv_name)
    id_test = df_test["ID"]
    Xb_test = df_test.iloc[:,9:]
    Xc_test = df_test.iloc[:,1:9]

    # string to numerical
    label_dict = defaultdict(LabelEncoder)
    pd.concat([Xc_train,Xc_test]).apply(lambda x: label_dict[x.name].fit(x.sort_values()))
    Xc_train = Xc_train.apply(lambda x: label_dict[x.name].transform(x))
    Xc_test = Xc_test.apply(lambda x: label_dict[x.name].transform(x))

    # remove outlier
    Xb_train = Xb_train[y_train < 200]
    Xc_train = Xc_train[y_train < 200]
    id_train = id_train[y_train < 200]
    y_train = y_train[y_train < 200]

    # check shapes
    if verbose >= 2:
        print("\tid_train shape: ", id_train.shape)
        print("\ty_train shape: ", y_train.shape)
        print("\tXc_train shape: ", Xc_train.shape)
        print("\tXb_train shape: ", Xb_train.shape)
        print("\tid_test shape: ", id_test.shape)
        print("\tXc_test shape: ", Xc_test.shape)
        print("\tXb_test shape: ", Xb_test.shape)


    ##### Make several cross-validation k-folds
    y_trainpred, y_traintest = [], []
    n_total = 3
    for ix_cv in range(n_total):

        ### Init cross-validation K-folds
        n_folds = 5
        cv = model_selection.KFold(n_splits=n_folds, shuffle=True)

        ### Split folds and fit+predict
        fold_cnt = 0
        for valtrain_index, valtest_index in cv.split(Xb_train.values):
            fold_cnt += 1
            if verbose >= 1: print("BIG CV: Processing fold number %d/%d..."%(fold_cnt+n_folds*ix_cv,n_folds*n_total))
            # split features and target labels
            id_valtrain, id_valtest = id_train.iloc[valtrain_index].values, id_train.iloc[valtest_index].values
            y_valtrain, y_valtest = y_train.iloc[valtrain_index].values, y_train.iloc[valtest_index].values
            Xb_valtrain, Xb_valtest = Xb_train.iloc[valtrain_index].values, Xb_train.iloc[valtest_index].values
            Xc_valtrain, Xc_valtest = Xc_train.iloc[valtrain_index], Xc_train.iloc[valtest_index]

            ##### Extract features
            if verbose >= 4: print("Extract features...")
            X_valtrain, X_valtest = [], []

            ### add binary features
            X_valtrain.append(Xb_valtrain)
            X_valtest.append(Xb_valtest)

            ### add categorical features
            X_valtrain.append(Xc_valtrain.values)
            X_valtest.append(Xc_valtest.values)

            ### encode categorical
            ohe = OneHotEncoder(handle_unknown='ignore')
            ohe.fit(Xc_valtrain)
            Xohe_valtrain = ohe.transform(Xc_valtrain).toarray()
            Xohe_valtest = ohe.transform(Xc_valtest).toarray()
            X_valtrain.append(Xohe_valtrain)
            X_valtest.append(Xohe_valtest)

            ### Add id
            Xid_valtrain = np.array([id_valtrain]).T
            Xid_valtest = np.array([id_valtest]).T
            X_valtrain.append(Xid_valtrain)
            X_valtest.append(Xid_valtest)

            ### PCA
            pca = PCA()
            pca.fit(np.hstack(X_valtrain))
            Xpca_valtrain = pca.transform(np.hstack(X_valtrain))
            Xpca_valtest = pca.transform(np.hstack(X_valtest))
            X_valtrain.append(Xpca_valtrain)
            X_valtest.append(Xpca_valtest)

            """### ICA
            ica = FastICA()
            ica.fit(np.hstack(X_valtrain))
            Xica_valtrain = ica.transform(np.hstack(X_valtrain))
            Xica_valtest = ica.transform(np.hstack(X_valtest))
            X_valtrain.append(Xica_valtrain)
            X_valtest.append(Xica_valtest)"""

            ##### Merge

            # merge all features
            X_valtrain = np.hstack(X_valtrain)
            X_valtest = np.hstack(X_valtest)

            # remove constant
            vt = VarianceThreshold()
            vt.fit(X_valtrain)
            X_valtrain = vt.transform(X_valtrain)
            X_valtest = vt.transform(X_valtest)

            """# drop correlations
            X_valtrain = pd.DataFrame(X_valtrain)
            X_valtest = pd.DataFrame(X_valtest)
            to_drop = drop_correlations(X_valtrain)
            X_valtrain = X_valtrain.drop(X_valtrain.columns[to_drop], axis=1).values
            X_valtest = X_valtest.drop(X_valtest.columns[to_drop], axis=1).values"""

            # select features
            n_jobs = 28
            selector = SelectFromModel(XGBRegressor(n_estimators=1120, objective='reg:linear', gamma=0, reg_lambda=1, min_child_weight=4,
                                       learning_rate=0.02, subsample=0.8, colsample_bytree=0.8, max_depth=4, nthread=n_jobs),
                                       threshold='1.25*median')
            selector.fit(X_valtrain)
            X_valtrain = selector.transform(X_valtrain)
            X_valtest = selector.transform(X_valtest)

            if verbose >= 3:
                print("\tX_valtrain shape: ", X_valtrain.shape)
                print("\tX_valtest shape: ", X_valtest.shape)



            ### Train model
            if verbose >= 4: print("Train model...")
            n_jobs = 28
            reg = XGBRegressor(n_estimators=1120, objective='reg:linear', gamma=0, reg_lambda=1, min_child_weight=4,
                               learning_rate=0.02, subsample=0.8, colsample_bytree=0.8, max_depth=4, nthread=n_jobs)
            reg.fit(X_valtrain, y_valtrain)
            #reg_list, reg_final = fit_stacked_regressors(X_valtrain, y_valtrain,
            #                      add_raw_features=True, verbose=verbose)

            ### Predict with model
            if verbose >= 4: print("Predict with model...")
            y_valpred = reg.predict(X_valtest)
            #y_valpred = predict_stacked_regressors(X_valtest, reg_list, reg_final,
            #            add_raw_features=True, verbose=verbose)

            ### Append preds and tests
            y_trainpred.extend(y_valpred)
            y_traintest.extend(y_valtest)


    ##### Compute R2-score
    r2_score = metrics.r2_score(y_traintest, y_trainpred)
    print("FINAL CV R2: ", r2_score)

    if verbose >= 1: print("Save predictions...")
    trainpred_csv_name = "../data/mercedes/trainpred.csv"
    trainpred_df = pd.DataFrame(np.array([y_trainpred]).T, columns=["y"])
    trainpred_df.to_csv(trainpred_csv_name, index=False)
    traintest_csv_name = "../data/mercedes/traintest.csv"
    traintest_df = pd.DataFrame(np.array([y_traintest]).T, columns=["y"])
    traintest_df.to_csv(traintest_csv_name, index=False)



    """### Save predictions
    if verbose >= 1: print("Save predictions...")
    pred_csv_name = "../data/mercedes/pred.csv"
    pred_df = pd.DataFrame(np.array([id_test, y_pred]).T, columns=["ID", "y"])
    pred_df["ID"] = pred_df["ID"].astype(int)
    pred_df.to_csv(pred_csv_name, index=False)"""







#==============================================
#                   Main
#==============================================
if __name__ == '__main__':
    main(3)
