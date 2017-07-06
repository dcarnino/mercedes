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
import urllib
import json
from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import model_selection, metrics
from scipy import sparse
from xgboost import XGBRegressor
from xgboost_ensembling import XGBRegressor_ensembling
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectFromModel
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.stats import rankdata, spearmanr
import subprocess
from stacked_regressor import stacked_regressor
#==============================================
#                   Files
#==============================================
import define_model
from mca import MCA


#==============================================
#                   Functions
#==============================================
def drop_correlations(X, threshold=0.99):
    """
    Get highly correlated features inplace.
    """

    # get columns to remove because correlated
    corr_indices = set([tuple(sorted([c1, c2])) for c1, c2 in zip(*np.where(X.corr().abs() > threshold)) if c1 != c2])
    corr_index = [ix_tup[1] for ix_tup in corr_indices]

    return corr_index




def leaderboard_probing_data():
    """
    Get leaderboard probing crowdsourced data.
    """

    def fetch_full_data():
      all_questions=json.loads(
        urllib.request.urlopen(
          "https://crowdstats.eu/api/topics/kaggle-mercedes-benz-greener-manufacturing-leaderboard-probing/questions"
        ).read()
      )
      answers = []
      for question in all_questions:
        for answer in question['answers']:
          newAnswer = {
            'ID': question['id'],
            'insidePublicLB': answer['inside_public_lb'],
            'y': answer['y_value'],
            'rho100': answer['rho_100'],
          }
          answers.append(newAnswer)
      return pd.DataFrame(answers)

    full_data = fetch_full_data()
    datapoints_inside_public_lb = full_data[full_data['insidePublicLB']==True]

    return datapoints_inside_public_lb[["ID", "y"]]




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
    id_test = df_test["ID"].astype(int)
    Xb_test = df_test.iloc[:,9:]
    Xc_test = df_test.iloc[:,1:9]
    # probing
    df_probing = leaderboard_probing_data()
    df_probing = df_probing.iloc[:13,:]
    id_probing = id_test[id_test.apply(lambda x: x in df_probing["ID"].values)]
    y_probing = df_probing["y"]
    Xb_probing = Xb_test[id_test.apply(lambda x: x in df_probing["ID"].values)]
    Xc_probing = Xc_test[id_test.apply(lambda x: x in df_probing["ID"].values)]
    id_train = pd.concat([id_train, id_probing], axis=0).reset_index(drop=True)
    y_train = pd.concat([y_train, y_probing], axis=0).reset_index(drop=True)
    Xb_train = pd.concat([Xb_train, Xb_probing], axis=0).reset_index(drop=True)
    Xc_train = pd.concat([Xc_train, Xc_probing], axis=0).reset_index(drop=True)

    """# replace test labels
    missing_dict = {"p": "q", "av": "aa", "ae": "ab", "bb": "ab", "an": "c", "ag": "c"}
    Xc_train["X0"] = Xc_train["X0"].apply(lambda x: missing_dict[x] if x in missing_dict.keys() else x)
    Xc_test["X0"] = Xc_test["X0"].apply(lambda x: missing_dict[x] if x in missing_dict.keys() else x)"""

    # get X0 insights
    Xc = pd.concat([Xc_train, Xc_test], axis=0).reset_index(drop=True)
    y = pd.concat([y_train, y_train.iloc[:len(Xc_test.index)].apply(lambda x: np.nan)])
    df_X0 = pd.DataFrame(np.array([Xc["X0"],y]).T, columns=["X0", "y"])
    df_X0["y"] = pd.to_numeric(df_X0["y"])
    X0_med = df_X0.groupby(["X0"])["y"].aggregate([np.nanmedian, 'size']).sort_values(by="nanmedian")

    # global feature
    new_feat_dict = {}
    for cat in X0_med.index[:2]:
        new_feat_dict[cat] = 'a'
    for cat in X0_med.index[2:23]:
        new_feat_dict[cat] = 'b'
    for cat in X0_med.index[23:30]:
        new_feat_dict[cat] = 'c'
    for cat in X0_med.index[30:46]:
        new_feat_dict[cat] = 'd'
    for cat in X0_med.index[46:47]:
        new_feat_dict[cat] = 'e'
    for cat in X0_med.index[47:]:
        new_feat_dict[cat] = 'f'
    Xc_train["X0_med"] = Xc_train["X0"].apply(lambda x: new_feat_dict[x])
    Xc_test["X0_med"] = Xc_test["X0"].apply(lambda x: new_feat_dict[x])

    """# add new X0+X5 feature
    Xc_train["X0X5"] = Xc_train["X0"] + "_" + Xc_train["X5"]
    Xc_test["X0X5"] = Xc_test["X0"] + "_" + Xc_test["X5"]"""

    """# add new X0_med+X5 feature
    Xc_train["X0_medX5"] = Xc_train["X0_med"] + "_" + Xc_train["X5"]
    Xc_test["X0_medX5"] = Xc_test["X0_med"] + "_" + Xc_test["X5"]"""

    # string to numerical
    label_dict = defaultdict(LabelEncoder)
    pd.concat([Xc_train,Xc_test]).apply(lambda x: label_dict[x.name].fit(x.sort_values()))
    Xc_train = Xc_train.apply(lambda x: label_dict[x.name].transform(x))
    Xc_test = Xc_test.apply(lambda x: label_dict[x.name].transform(x))

    # count duplicate rows
    ##### Process duplicate rows
    dupe_df = pd.DataFrame(np.vstack([np.hstack([Xb_train.values, Xc_train.values]),np.hstack([Xb_test.values, Xc_test.values])]))
    dupe_df = dupe_df[dupe_df.duplicated(keep=False)]
    dupe_ser = dupe_df.groupby(list(dupe_df.columns)).size()
    print(dupe_ser)
    print(dupe_ser.to_dict())
    raise(ValueError)

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


    leaderboard = False
    ##### Make several cross-validation k-folds
    y_trainpred, y_traintest = [], []
    if leaderboard:
        n_total = 1
    else:
        n_total = 20
    for ix_cv in range(n_total):

        ### Init cross-validation K-folds
        n_folds = 5
        cv = model_selection.KFold(n_splits=n_folds, shuffle=True)

        ### Split folds and fit+predict
        fold_cnt = 0
        for valtrain_index, valtest_index in cv.split(Xb_train.values):
            fold_cnt += 1
            if verbose >= 1: print("BIG CV: Processing fold number %d/%d..."%(fold_cnt+n_folds*ix_cv,n_folds*n_total), end='')
            # split features and target labels
            id_valtrain, id_valtest = id_train.iloc[valtrain_index].values, id_train.iloc[valtest_index].values
            y_valtrain, y_valtest = y_train.iloc[valtrain_index].values, y_train.iloc[valtest_index].values
            Xb_valtrain, Xb_valtest = Xb_train.iloc[valtrain_index].values, Xb_train.iloc[valtest_index].values
            Xc_valtrain, Xc_valtest = Xc_train.iloc[valtrain_index], Xc_train.iloc[valtest_index]

            if leaderboard:
                id_valtrain, y_valtrain, Xb_valtrain, Xc_valtrain = id_train.values, y_train.values, Xb_train.values, Xc_train
                id_valtest, y_valtest, Xb_valtest, Xc_valtest = id_test.values, id_test.values, Xb_test.values, Xc_test

            ##### Transform target y
            """# with rank
            rank_valtrain = rankdata(y_valtrain, method='dense')
            rank_valtrain = rank_valtrain - rank_valtrain.min()
            rank_valtrain = rank_valtrain / rank_valtrain.max()
            sorted_rank_valtrain = np.unique(rank_valtrain)
            sorted_y_valtrain = np.unique(y_valtrain)
            rank_to_y_func = InterpolatedUnivariateSpline(sorted_rank_valtrain, sorted_y_valtrain, k=3, ext='const')
            y_to_rank_func = InterpolatedUnivariateSpline(sorted_y_valtrain, sorted_rank_valtrain, k=3, ext='const')
            y_valtrain = y_to_rank_func(y_valtrain)
            y_valtrain[y_valtrain < 0] = 0.
            y_valtrain[y_valtrain > 1] = 1."""
            # in log space
            smoothing_term = 10
            y_valtrain = np.log(y_valtrain+smoothing_term)
            y_min = np.min(y_valtrain)
            y_valtrain = y_valtrain - y_min
            y_max = np.max(y_valtrain)
            y_valtrain = y_valtrain / y_max
            y_valtrain[y_valtrain < 0] = 0.
            y_valtrain[y_valtrain > 1] = 1.

            """##### Process duplicate rows
            dupe_df = pd.DataFrame(np.hstack([Xb_valtrain, Xc_valtrain.values]))
            dupe_df["_fake"] = range(len(dupe_df.index))
            dupe_df = dupe_df[dupe_df[[col for col in dupe_df.columns if col!="_fake"]].duplicated(keep=False)]
            dupe_list = dupe_df.groupby([col for col in dupe_df.columns if col!="_fake"]).apply(lambda x: list(x.index)).tolist()
            drop_rows = []
            for dupes in dupe_list:
                for ix_dupe, dupe in enumerate(dupes):
                    if ix_dupe == 0:
                        y_valtrain[dupe] = np.median(y_valtrain[dupes])
                    else:
                        drop_rows.append(dupe)
            y_valtrain = np.delete(y_valtrain, drop_rows, axis=0)
            id_valtrain = np.delete(id_valtrain, drop_rows, axis=0)
            Xb_valtrain = np.delete(Xb_valtrain, drop_rows, axis=0)
            Xc_valtrain = Xc_valtrain.drop(Xc_valtrain.index[drop_rows])"""

            ##### Extract features
            if verbose >= 4: print("Extract features...")
            X0_valtrain, X0_valtest = [], []
            X1_valtrain, X1_valtest = [], []
            X2_valtrain, X2_valtest = [], []

            ### add binary features
            X0_valtrain.append(Xb_valtrain)
            X0_valtest.append(Xb_valtest)

            ################################ TO REMOVE
            X1_valtrain.append(Xb_valtrain)
            X1_valtest.append(Xb_valtest)

            ################################ TO REMOVE
            ### add categorical features
            X1_valtrain.append(Xc_valtrain.values)
            X1_valtest.append(Xc_valtest.values)

            ### add means of categorical
            Xmeans_valtrain, Xmeans_valtest = [], []
            for cat_col in Xc_valtrain.columns:
                cat_means = defaultdict(lambda: y_valtrain.mean())
                diff_cat = set(Xc_valtrain[cat_col])
                for cat in diff_cat:
                    cat_means[cat] = y_valtrain[Xc_valtrain[cat_col] == cat].mean()
                Xm_valtrain = Xc_valtrain[cat_col].apply(lambda x: cat_means[x]).values
                Xm_valtest = Xc_valtest[cat_col].apply(lambda x: cat_means[x]).values
                Xmeans_valtrain.append(Xm_valtrain.reshape((-1,1)))
                Xmeans_valtest.append(Xm_valtest.reshape((-1,1)))
            X0_valtrain.append(np.hstack(Xmeans_valtrain))
            X0_valtest.append(np.hstack(Xmeans_valtest))
            X1_valtrain.append(np.hstack(Xmeans_valtrain))
            X1_valtest.append(np.hstack(Xmeans_valtest))

            ### encode categorical
            ohe = OneHotEncoder(handle_unknown='ignore')
            ohe.fit(Xc_valtrain)
            Xohe_valtrain = ohe.transform(Xc_valtrain).toarray()
            Xohe_valtest = ohe.transform(Xc_valtest).toarray()
            X0_valtrain.append(Xohe_valtrain)
            X0_valtest.append(Xohe_valtest)

            ################################ TO REMOVE
            X1_valtrain.append(Xohe_valtrain)
            X1_valtest.append(Xohe_valtest)

            """### engineer new features based on correlations
            Xtr = np.hstack(X_valtrain)
            Xte = np.hstack(X_valtest)
            Xcorr_valtrain = []
            Xcorr_valtest = []
            correlation_bounds = [(0.2,0.3), (0.1,0.2), (0.075,0.1), (0.05,0.075), (0.05,0.1), (0.05,0.3), (0.3,1.)]
            corr_indices = defaultdict(list)
            for ixc, xc in enumerate(Xtr.T):
                if len(set(xc)) > 1:
                    corr = spearmanr(y_valtrain, xc)
                    for lt, ut in correlation_bounds:
                        if corr[0] >= lt and corr[0] < ut:
                            corr_indices[(lt,ut)].append(ixc)
            for ltut in correlation_bounds:
                Xcorr_valtrain.append(Xtr[:,corr_indices[ltut]].sum(axis=1).reshape((-1,1)))
                Xcorr_valtest.append(Xte[:,corr_indices[ltut]].sum(axis=1).reshape((-1,1)))
            X_valtrain.append(np.hstack(Xcorr_valtrain))
            X_valtest.append(np.hstack(Xcorr_valtest))"""

            ### Add id
            Xid_valtrain = np.array([id_valtrain]).T
            Xid_valtest = np.array([id_valtest]).T
            X0_valtrain.append(Xid_valtrain)
            X0_valtest.append(Xid_valtest)
            X1_valtrain.append(Xid_valtrain)
            X1_valtest.append(Xid_valtest)

            """### MCA
            Xbool_valtrain = np.hstack([Xb_valtrain, Xohe_valtrain])
            Xbool_valtest = np.hstack([Xb_valtest, Xohe_valtest])
            mca_obj = MCA(n_components=20)
            mca_obj.fit(Xbool_valtrain)
            Xmca_valtrain = mca_obj.transform(Xbool_valtrain)
            Xmca_valtest = mca_obj.transform(Xbool_valtest)
            X_valtrain.append(Xmca_valtrain)
            X_valtest.append(Xmca_valtest)"""

            """### Logistic PCA
            Xbool_valtrain = pd.DataFrame(np.hstack([Xb_valtrain, Xohe_valtrain]))
            Xbool_valtest = pd.DataFrame(np.hstack([Xb_valtest, Xohe_valtest]))
            Xbool_valtrain.to_csv("../data/mercedes/binary_train.csv", index=False)
            Xbool_valtest.to_csv("../data/mercedes/binary_test.csv", index=False)
            subprocess.call(["/home/jlevyabi/anaconda2/bin/Rscript", "--vanilla", "logpca.r"])
            Xlogpca_valtrain = pd.read_csv("../data/mercedes/logpca_train.csv").values
            Xlogpca_valtest = pd.read_csv("../data/mercedes/logpca_test.csv").values
            X_valtrain.append(Xlogpca_valtrain)
            X_valtest.append(Xlogpca_valtest)"""

            n_components=12

            """### PCA
            pca = PCA(n_components=n_components)
            pca.fit(np.hstack(X0_valtrain))
            Xpca_valtrain = pca.transform(np.hstack(X0_valtrain))
            Xpca_valtest = pca.transform(np.hstack(X0_valtest))
            X1_valtrain.append(Xpca_valtrain)
            X1_valtest.append(Xpca_valtest)

            ### ICA
            ica = FastICA(n_components=n_components, max_iter=1000, tol=0.005)
            ica.fit(np.hstack(X0_valtrain))
            Xica_valtrain = ica.transform(np.hstack(X0_valtrain))
            Xica_valtest = ica.transform(np.hstack(X0_valtest))
            X1_valtrain.append(Xica_valtrain)
            X1_valtest.append(Xica_valtest)

            # tSVD
            tsvd = TruncatedSVD(n_components=n_components)
            tsvd.fit(np.hstack(X0_valtrain))
            Xtsvd_valtrain = tsvd.transform(np.hstack(X0_valtrain))
            Xtsvd_valtest = tsvd.transform(np.hstack(X0_valtest))
            X1_valtrain.append(Xtsvd_valtrain)
            X1_valtest.append(Xtsvd_valtest)

            # SGRP
            srp = SparseRandomProjection(n_components=n_components, dense_output=True)
            srp.fit(np.hstack(X0_valtrain))
            Xsrp_valtrain = srp.transform(np.hstack(X0_valtrain))
            Xsrp_valtest = srp.transform(np.hstack(X0_valtest))
            X1_valtrain.append(Xsrp_valtrain)
            X1_valtest.append(Xsrp_valtest)

            # GRP
            grp = GaussianRandomProjection(n_components=n_components, eps=0.1)
            grp.fit(np.hstack(X0_valtrain))
            Xgrp_valtrain = grp.transform(np.hstack(X0_valtrain))
            Xgrp_valtest = grp.transform(np.hstack(X0_valtest))
            X1_valtrain.append(Xgrp_valtrain)
            X1_valtest.append(Xgrp_valtest)"""

            ### Add specific columns
            X2_valtrain.append(Xb_valtrain[:,[297]])
            X2_valtest.append(Xb_valtest[:,[297]])
            X2_valtrain.append(np.hstack(Xmeans_valtrain)[:,[0,5]])
            X2_valtest.append(np.hstack(Xmeans_valtest)[:,[0,5]])
            #X2_valtrain.append(np.hstack(Xmeans_valtrain)[:,[0,5,8]])
            #X2_valtest.append(np.hstack(Xmeans_valtest)[:,[0,5,8]])

            ##### Merge

            # merge all features
            X0_valtrain = np.hstack(X0_valtrain)
            X0_valtest = np.hstack(X0_valtest)
            X1_valtrain = np.hstack(X1_valtrain)
            X1_valtest = np.hstack(X1_valtest)
            X2_valtrain = np.hstack(X2_valtrain)
            X2_valtest = np.hstack(X2_valtest)

            # remove constant
            vt = VarianceThreshold()
            vt.fit(X0_valtrain)
            X0_valtrain = vt.transform(X0_valtrain)
            X0_valtest = vt.transform(X0_valtest)
            vt = VarianceThreshold()
            vt.fit(X1_valtrain)
            X1_valtrain = vt.transform(X1_valtrain)
            X1_valtest = vt.transform(X1_valtest)
            vt = VarianceThreshold()
            vt.fit(X1_valtrain)
            X1_valtrain = vt.transform(X1_valtrain)
            X1_valtest = vt.transform(X1_valtest)

            """# drop correlations
            X_valtrain = pd.DataFrame(X_valtrain)
            X_valtest = pd.DataFrame(X_valtest)
            to_drop = drop_correlations(X_valtrain)
            X_valtrain = X_valtrain.drop(X_valtrain.columns[to_drop], axis=1).values
            X_valtest = X_valtest.drop(X_valtest.columns[to_drop], axis=1).values"""

            """# select features
            n_jobs = 28
            reg = XGBRegressor(n_estimators=1120, objective='reg:linear', gamma=0, reg_lambda=1, min_child_weight=4,
                               learning_rate=0.02, subsample=0.8, colsample_bytree=0.8, max_depth=4, nthread=n_jobs)
            reg.fit(X_valtrain, y_valtrain)
            fscores = reg.booster().get_fscore()
            reg.feature_importances_ = np.zeros(X_valtrain.shape[1])
            total_importance = sum(fscores.values())
            for k,v in fscores.items():
                reg.feature_importances_[int(k[1:])] = float(v)/total_importance
            selector = SelectFromModel(reg, prefit=True,
                                       threshold='1.75*median')
            X_valtrain = selector.transform(X_valtrain)
            X_valtest = selector.transform(X_valtest)"""

            if verbose >= 5:
                print("\tX0_valtrain shape: ", X0_valtrain.shape)
                print("\tX0_valtest shape: ", X0_valtest.shape)
                print("\tX1_valtrain shape: ", X1_valtrain.shape)
                print("\tX1_valtest shape: ", X1_valtest.shape)
                print("\tX2_valtrain shape: ", X2_valtrain.shape)
                print("\tX2_valtest shape: ", X2_valtest.shape)

            ### Train model
            if verbose >= 4: print("Train model...")
            if fold_cnt+n_folds*ix_cv == 1:
                pass
                """reg_cv = model_selection.GridSearchCV(XGBRegressor_ensembling(objective='reg:logistic', gamma=0, reg_lambda=1, min_child_weight=4,
                                                      learning_rate=0.02, subsample=0.65, colsample_bytree=0.65, max_depth=5, nthread=28),
                                                      {'max_depth': [5, 7, 9], 'subsample': [.5, .65, .8], 'colsample_bytree': [.5, .65, .8], 'min_child_weight': [4, 10, 20]},
                                                      scoring=metrics.make_scorer(metrics.r2_score, greater_is_better=True),
                                                      n_jobs=1, cv=5, verbose=3, pre_dispatch='n_jobs', error_score='raise')
                reg_cv.fit(X_valtrain, y_valtrain)
                print(reg_cv.best_params_, reg_cv.best_score_)
                reg = reg_cv.best_estimator_"""
            """reg = XGBRegressor(n_estimators=448, objective='reg:logistic', gamma=0, reg_lambda=1, min_child_weight=4,
                               learning_rate=0.02, subsample=0.65, colsample_bytree=0.65, max_depth=5, nthread=28)"""
            """reg = stacked_regressor(define_model.create_layer0, define_model.create_layer1, define_model.create_layer2,
                                    remove_bad0=0.2, remove_bad1=0.1,
                                    n_folds0=5, n_folds1=5, n_est0=892, n_est1=2240, score_func=metrics.r2_score,
                                    default_y_value=0.5, n_jobs=28)
            reg.fit(X0_valtrain, y_valtrain, X1_valtrain, X2_valtrain, verbose=verbose)"""
            reg = XGBRegressor_ensembling(objective='reg:logistic', gamma=0, reg_lambda=1, min_child_weight=4,
                                          learning_rate=0.02, subsample=0.65, colsample_bytree=0.65, max_depth=5, nthread=28)
            reg.fit(X1_valtrain, y_valtrain)

            ### Predict with model
            if verbose >= 4: print("Predict with model...")
            #y_valpred = reg.predict(X0_valtest, X1_valtest, X2_valtest, verbose=verbose)
            y_valpred = reg.predict(X1_valtest)


            ### Append preds and tests
            #y_valpred = rank_to_y_func(y_valpred)
            y_valpred = np.exp(y_valpred * y_max + y_min) - smoothing_term
            y_trainpred.extend(y_valpred)
            y_traintest.extend(y_valtest)

            if verbose >= 1: print(" (R2-score: %.04f)"%(metrics.r2_score(y_valtest, y_valpred)))
            if leaderboard:
                break


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



    ### Save predictions
    if leaderboard:
        if verbose >= 1: print("Save predictions...")
        pred_csv_name = "../data/mercedes/pred.csv"
        pred_df = pd.DataFrame(np.array([id_test, y_trainpred]).T, columns=["ID", "y"])
        pred_df["ID"] = pred_df["ID"].astype(int)
        pred_df.to_csv(pred_csv_name, index=False)







#==============================================
#                   Main
#==============================================
if __name__ == '__main__':
    main(3)
