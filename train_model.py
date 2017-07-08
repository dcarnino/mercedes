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
pd.options.mode.chained_assignment = None  # default='warn'
import urllib
import json
from operator import itemgetter
from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import ExtraTreesRegressor, BaggingRegressor, RandomForestRegressor
from sklearn import model_selection, metrics
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import Ridge
from scipy import sparse
from xgboost import XGBRegressor
from xgboost_ensembling import XGBRegressor_ensembling, XGBClassifier_ensembling
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.stats import rankdata, spearmanr
import subprocess
from stacked_regressor import stacked_regressor
from decorrelating_estimator import correlation_ensembling
import lightgbm as lgb
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
    """df_probing = leaderboard_probing_data()
    df_probing = df_probing.iloc[:13,:]
    id_probing = id_test[id_test.apply(lambda x: x in df_probing["ID"].values)]
    y_probing = df_probing["y"]
    Xb_probing = Xb_test[id_test.apply(lambda x: x in df_probing["ID"].values)]
    Xc_probing = Xc_test[id_test.apply(lambda x: x in df_probing["ID"].values)]
    id_train = pd.concat([id_train, id_probing], axis=0).reset_index(drop=True)
    y_train = pd.concat([y_train, y_probing], axis=0).reset_index(drop=True)
    Xb_train = pd.concat([Xb_train, Xb_probing], axis=0).reset_index(drop=True)
    Xc_train = pd.concat([Xc_train, Xc_probing], axis=0).reset_index(drop=True)"""

    """# replace test labels
    missing_dict = {"p": "q", "av": "ac", "ae": "d", "bb": "aw", "an": "c", "ag": "c"}
    Xc_train["X0"] = Xc_train["X0"].apply(lambda x: missing_dict[x] if x in missing_dict.keys() else x)
    Xc_test["X0"] = Xc_test["X0"].apply(lambda x: missing_dict[x] if x in missing_dict.keys() else x)"""

    ### get Xx insights
    Xc = pd.concat([Xc_train, Xc_test], axis=0).reset_index(drop=True)
    y = pd.concat([y_train, y_train.iloc[:len(Xc_test.index)].apply(lambda x: np.nan)])

    # get X0 insights
    df_X0 = pd.DataFrame(np.array([Xc["X0"],y]).T, columns=["X0", "y"])
    df_X0["y"] = pd.to_numeric(df_X0["y"])
    X0_med = df_X0.groupby(["X0"])["y"].aggregate([np.nanmedian, 'size']).sort_values(by="nanmedian")
    # X0 grouped feature
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

    # get X5 insights
    df_X5 = pd.DataFrame(np.array([Xc["X5"],y]).T, columns=["X5", "y"])
    df_X5["y"] = pd.to_numeric(df_X5["y"])
    X5_med = df_X5.groupby(["X5"])["y"].aggregate([np.nanmedian, 'size']).sort_values(by="nanmedian")
    # X5 grouped feature
    new_feat_dict = {}
    for cat in X5_med.index[:2]:
        new_feat_dict[cat] = 'a'
    for cat in X5_med.index[2:6]:
        new_feat_dict[cat] = 'b'
    for cat in X5_med.index[6:29]:
        new_feat_dict[cat] = 'c'
    for cat in X5_med.index[29:30]:
        new_feat_dict[cat] = 'd'
    for cat in X5_med.index[30:]:
        new_feat_dict[cat] = 'e'
    Xc_train["X5_med"] = Xc_train["X5"].apply(lambda x: new_feat_dict[x])
    Xc_test["X5_med"] = Xc_test["X5"].apply(lambda x: new_feat_dict[x])

    """# add new X0+X5 feature
    Xc_train["X0X5"] = Xc_train["X0"] + "_" + Xc_train["X5"]
    Xc_test["X0X5"] = Xc_test["X0"] + "_" + Xc_test["X5"]"""

    """# add new X0_med+X5 feature
    Xc_train["X0_medX5"] = Xc_train["X0_med"] + "_" + Xc_train["X5"]
    Xc_test["X0_medX5"] = Xc_test["X0_med"] + "_" + Xc_test["X5"]"""

    """# add new X0_med+X5_med feature
    Xc_train["X0_medX5_med"] = Xc_train["X0_med"] + "_" + Xc_train["X5_med"]
    Xc_test["X0_medX5_med"] = Xc_test["X0_med"] + "_" + Xc_test["X5_med"]"""

    # string to numerical
    label_dict = defaultdict(LabelEncoder)
    pd.concat([Xc_train,Xc_test]).apply(lambda x: label_dict[x.name].fit(x.sort_values()))
    Xc_train = Xc_train.apply(lambda x: label_dict[x.name].transform(x))
    Xc_test = Xc_test.apply(lambda x: label_dict[x.name].transform(x))

    # count duplicate rows
    ##### Process duplicate rows
    dupe_df = pd.DataFrame(np.vstack([np.hstack([Xb_train.values, Xc_train.values]),np.hstack([Xb_test.values, Xc_test.values])]))
    dupe_ser = dupe_df.groupby(list(dupe_df.columns)).size()
    dupe_dict = dupe_ser.to_dict()
    dupe_count_train = pd.Series([dupe_dict[dupe_tuple] for dupe_tuple in pd.DataFrame(np.hstack([Xb_train.values, Xc_train.values])).itertuples(index=False)]).rename("dupe_count")
    dupe_count_test = pd.Series([dupe_dict[dupe_tuple] for dupe_tuple in pd.DataFrame(np.hstack([Xb_test.values, Xc_test.values])).itertuples(index=False)]).rename("dupe_count")

    ### EMA
    alpha = .3
    Xc = pd.concat([Xc_train, Xc_test], axis=0).reset_index(drop=True)
    Xb = pd.concat([Xb_train, Xb_test], axis=0).reset_index(drop=True)
    id_ = pd.concat([id_train, id_test])
    sort_mask = np.argsort(id_.values)
    inverse_sort_mask = np.argsort(sort_mask)
    sorted_Xc = Xc.iloc[sort_mask,:].values
    sorted_Xb = Xb.iloc[sort_mask,:].values
    for ix, (rowc, rowb) in enumerate(zip(sorted_Xc, sorted_Xb)):
        if ix == 0:
            curc = rowc
            curb = rowb
        else:
            curc = alpha * rowc + (1 - alpha) * curc
            curb = alpha * rowb + (1 - alpha) * curb
        sorted_Xc[ix,:] = curc
        sorted_Xb[ix,:] = curb
    Xemac = sorted_Xc[inverse_sort_mask, :]
    Xemab = sorted_Xb[inverse_sort_mask, :]
    Xemac_train = pd.DataFrame(Xemac[:len(Xc_train), :])
    Xemac_test = pd.DataFrame(Xemac[len(Xc_train):, :])
    Xemab_train = pd.DataFrame(Xemab[:len(Xb_train), :])
    Xemab_test = pd.DataFrame(Xemab[len(Xb_train):, :])

    # remove outlier
    Xb_train = Xb_train[y_train < 200]
    Xc_train = Xc_train[y_train < 200]
    Xemab_train = Xemab_train[y_train < 200]
    Xemac_train = Xemac_train[y_train < 200]
    id_train = id_train[y_train < 200]
    dupe_count_train = dupe_count_train[y_train < 200]
    y_train = y_train[y_train < 200]

    # check shapes
    if verbose >= 2:
        print("\tid_train shape: ", id_train.shape)
        print("\tdupe_count_train shape: ", dupe_count_train.shape)
        print("\ty_train shape: ", y_train.shape)
        print("\tXc_train shape: ", Xc_train.shape)
        print("\tXb_train shape: ", Xb_train.shape)
        print("\tXemac_train shape: ", Xemac_train.shape)
        print("\tXemab_train shape: ", Xemab_train.shape)
        print("\tid_test shape: ", id_test.shape)
        print("\tdupe_count_test shape: ", dupe_count_test.shape)
        print("\tXemac_test shape: ", Xemac_test.shape)
        print("\tXemab_test shape: ", Xemab_test.shape)

    # outliers
    outlier_train = defaultdict(list)
    outlier_test = defaultdict(list)

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
            dupe_count_valtrain, dupe_count_valtest = dupe_count_train.iloc[valtrain_index].values, dupe_count_train.iloc[valtest_index].values
            y_valtrain, y_valtest = y_train.iloc[valtrain_index].values, y_train.iloc[valtest_index].values
            Xb_valtrain, Xb_valtest = Xb_train.iloc[valtrain_index].values, Xb_train.iloc[valtest_index].values
            Xc_valtrain, Xc_valtest = Xc_train.iloc[valtrain_index], Xc_train.iloc[valtest_index]
            Xemab_valtrain, Xemab_valtest = Xemab_train.iloc[valtrain_index].values, Xemab_train.iloc[valtest_index].values
            Xemac_valtrain, Xemac_valtest = Xemac_train.iloc[valtrain_index].values, Xemac_train.iloc[valtest_index].values

            if leaderboard:
                id_valtrain, dupe_count_valtrain, y_valtrain, Xb_valtrain, Xc_valtrain, Xemab_valtrain, Xemac_valtrain = id_train.values, dupe_count_train.values, y_train.values, Xb_train.values, Xc_train, Xemab_train, Xemac_train
                id_valtest, dupe_count_valtest, y_valtest, Xb_valtest, Xc_valtest, Xemab_valtest, Xemac_valtest = id_test.values, dupe_count_test.values, id_test.values, Xb_test.values, Xc_test, Xemab_test, Xemac_test

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
                        y_valtrain[dupe] = np.mean(y_valtrain[dupes])
                    else:
                        drop_rows.append(dupe)
            y_valtrain = np.delete(y_valtrain, drop_rows, axis=0)
            id_valtrain = np.delete(id_valtrain, drop_rows, axis=0)
            dupe_count_valtrain = np.delete(dupe_count_valtrain, drop_rows, axis=0)
            Xb_valtrain = np.delete(Xb_valtrain, drop_rows, axis=0)
            Xc_valtrain = Xc_valtrain.drop(Xc_valtrain.index[drop_rows])"""

            """##### Replace test X0s not in train X0s

            ### save X0 to X0_med mapping
            Xc = pd.concat([Xc_valtrain, Xc_valtest], axis=0).reset_index(drop=True)
            X0_to_X0_med_mapping = {}
            for key, val in zip(Xc["X0"], Xc["X0_med"]):
                X0_to_X0_med_mapping[key] = val

            ### get training set
            # means
            Xmeans_valtrain, Xmeans_valtest = [], []
            for cat_col in Xc_valtrain.columns[1:]:
                cat_means = defaultdict(lambda: y_valtrain.mean())
                diff_cat = set(Xc_valtrain[cat_col])
                for cat in diff_cat:
                    cat_means[cat] = y_valtrain[Xc_valtrain[cat_col] == cat].mean()
                Xm_valtrain = Xc_valtrain[cat_col].apply(lambda x: cat_means[x]).values
                Xm_valtest = Xc_valtest[cat_col].apply(lambda x: cat_means[x]).values
                Xmeans_valtrain.append(Xm_valtrain.reshape((-1,1)))
                Xmeans_valtest.append(Xm_valtest.reshape((-1,1)))
            Xmeans_valtrain = np.hstack(Xmeans_valtrain)
            Xmeans_valtest = np.hstack(Xmeans_valtest)
            # ohe
            ohe = OneHotEncoder(handle_unknown='ignore')
            ohe.fit(Xc_valtrain.values[:,1:])
            Xohe_valtrain = ohe.transform(Xc_valtrain.values[:,1:]).toarray()
            Xohe_valtest = ohe.transform(Xc_valtest.values[:,1:]).toarray()
            # id
            Xid_valtrain = np.array([id_valtrain]).T
            Xid_valtest = np.array([id_valtest]).T
            # merge
            X_impute_traintrain = np.hstack([Xc_valtrain.values[:,1:], Xohe_valtrain, Xmeans_valtrain, Xb_valtrain, Xid_valtrain])
            X_impute_traintest = np.hstack([Xc_valtest.values[:,1:], Xohe_valtest, Xmeans_valtest, Xb_valtest, Xid_valtest])

            ### get target values
            X0_train_labels = set(Xc_valtrain.values[:,0])
            X0_test_labels = set(Xc_valtest.values[:,0])
            X0_missing_labels = X0_test_labels - X0_train_labels
            if len(X0_missing_labels) > 0:
                mask_test = np.array([xc in X0_missing_labels for xc in Xc_valtest.values[:,0]])
                # truely split train test
                y_impute_train = np.hstack([Xc_valtrain.values[:,0], Xc_valtest.values[:,0][~mask_test]])
                X_impute_train = np.vstack([X_impute_traintrain, X_impute_traintest[~mask_test]])
                X_impute_test = X_impute_traintest[mask_test]

                # encode labels
                le = LabelEncoder()
                y_impute_train = le.fit_transform(y_impute_train)

                ### classify
                clf = XGBClassifier_ensembling(objective='multi:softmax', gamma=0, reg_lambda=1, min_child_weight=4,
                                              learning_rate=0.02, subsample=0.65, colsample_bytree=0.65, max_depth=5, nthread=28)
                clf.fit(X_impute_train, y_impute_train)
                y_impute_pred = clf.predict(X_impute_test)
                y_impute_pred = le.inverse_transform(y_impute_pred)

                ### replace missing values with predicted values
                Xc_valtest.iloc[:,0][mask_test] = y_impute_pred

                ### change X0 med value based on new value
                Xc_valtest.loc[:,"X0_med"] = Xc_valtest.loc[:,"X0"].apply(lambda x: X0_to_X0_med_mapping[x])"""


            ##### Extract features
            if verbose >= 4: print("Extract features...")
            X0_valtrain, X0_valtest = [], []
            X1_valtrain, X1_valtest = [], []
            X2_valtrain, X2_valtest = [], []

            ### add categorical features
            X0_valtrain.append(Xc_valtrain.values)
            X0_valtest.append(Xc_valtest.values)
            X1_valtrain.append(Xc_valtrain.values)
            X1_valtest.append(Xc_valtest.values)

            ### add binary features
            X0_valtrain.append(Xb_valtrain)
            X0_valtest.append(Xb_valtest)
            X1_valtrain.append(Xb_valtrain)
            X1_valtest.append(Xb_valtest)

            """### add ema features
            X0_valtrain.append(Xemab_valtrain)
            X0_valtest.append(Xemab_valtest)
            X1_valtrain.append(Xemab_valtrain)
            X1_valtest.append(Xemab_valtest)
            X0_valtrain.append(Xemac_valtrain)
            X0_valtest.append(Xemac_valtest)
            X1_valtrain.append(Xemac_valtrain)
            X1_valtest.append(Xemac_valtest)"""

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
            """Xmeans_valtrain.append(Xmeans_valtrain[0]*Xmeans_valtrain[5])
            Xmeans_valtest.append(Xmeans_valtest[0]*Xmeans_valtest[5])
            Xmeans_valtrain.append(Xmeans_valtrain[0]+Xmeans_valtrain[5])
            Xmeans_valtest.append(Xmeans_valtest[0]+Xmeans_valtest[5])"""
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

            ### Add dupe_count
            Xdpc_valtrain = np.array([dupe_count_valtrain]).T
            Xdpc_valtest = np.array([dupe_count_valtest]).T
            X0_valtrain.append(Xdpc_valtrain)
            X0_valtest.append(Xdpc_valtest)
            X1_valtrain.append(Xdpc_valtrain)
            X1_valtest.append(Xdpc_valtest)

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
            #X2_valtrain.append(Xb_valtrain[:,[297]])
            #X2_valtest.append(Xb_valtest[:,[297]])
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

            """### replace if all the same
            X0_valtrain = X1_valtrain
            X0_valtest = X1_valtest
            X2_valtrain = X1_valtrain
            X2_valtest = X1_valtest"""

            if verbose >= 5:
                print("\tX0_valtrain shape: ", X0_valtrain.shape)
                print("\tX0_valtest shape: ", X0_valtest.shape)
                print("\tX1_valtrain shape: ", X1_valtrain.shape)
                print("\tX1_valtest shape: ", X1_valtest.shape)
                print("\tX2_valtrain shape: ", X2_valtrain.shape)
                print("\tX2_valtest shape: ", X2_valtest.shape)


            ### Define prior
            gmm_prior = None
            """gmm = GaussianMixture(n_components=5, n_init=100)
            gmm.fit(y_valtrain.reshape((-1, 1)))
            def gmm_prior(y_pred):
                y_pred = y_pred.reshape((-1, 1))
                y_score = gmm.score_samples(y_pred)
                y_score = y_score + y_score.min()
                #y_score = y_score.max() - y_score
                return y_score"""


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
            """X1_valtrain = None
            X1_valtest = None
            reg = stacked_regressor(define_model.create_layer0, define_model.create_layer1, define_model.create_layer2,
                                    combine_features_models=True, combine_features=True, combine_models=False,
                                    remove_bad0=0.2, remove_bad1=0.1,
                                    n_folds0=5, n_folds1=5, n_est0=892, n_est1=2240, score_func=metrics.r2_score,
                                    default_y_value=0.5, n_jobs=28)
            reg.fit(X0_valtrain, y_valtrain, X1_valtrain, X2_valtrain, verbose=verbose)"""
            #reg = XGBRegressor_ensembling(prior=gmm_prior, objective='reg:logistic', gamma=0, reg_lambda=1, min_child_weight=4,
            #                              learning_rate=0.02, subsample=0.65, colsample_bytree=0.65, max_depth=5, nthread=28)
            #reg = BaggingRegressor(base_estimator=reg, n_estimators=5)
            reg = XGBRegressor(n_estimators=224, objective='reg:logistic', gamma=0, reg_lambda=1, min_child_weight=4,
                                          learning_rate=0.02, subsample=0.65, colsample_bytree=0.65, max_depth=5, nthread=28)
            reg.fit(X1_valtrain, y_valtrain)
            """n_est = 500
            estimator_list = [XGBRegressor_ensembling(objective='reg:logistic', gamma=0, reg_lambda=1, min_child_weight=4,
                                                      learning_rate=0.02, subsample=0.65, colsample_bytree=0.65, max_depth=5, nthread=28) for ix in range(n_est)]
            reg = correlation_ensembling(estimator_list, keep_p=0.05)
            reg.fit(X1_valtrain, y_valtrain)"""

            ### Predict with model
            if verbose >= 4: print("Predict with model...")
            #y_valpred = reg.predict(X0_valtest, X1_valtest, X2_valtest, verbose=verbose)
            y_valpred = reg.predict(X1_valtest)

            """### Evenize the output based on rank
            rank_valpred = rankdata(y_valpred, method='dense')
            rank_valpred = rank_valpred - rank_valpred.min()
            rank_valpred = rank_valtrain / rank_valpred.max()
            y_valpred = rank_valpred"""

            ### Inverse transform y
            #y_valpred = rank_to_y_func(y_valpred)
            y_valpred = np.exp(y_valpred * y_max + y_min) - smoothing_term

            ### Rectify values manually
            fac = 2.
            """y_valpred[(y_valpred < 72) & (y_valpred > 30)] = y_valpred[(y_valpred < 72) & (y_valpred > 30)] + 3.*fac
            y_valpred[(y_valpred < 83) & (y_valpred > 80)] = y_valpred[(y_valpred < 83) & (y_valpred > 80)] - 2.*fac
            y_valpred[(y_valpred < 86) & (y_valpred > 83)] = y_valpred[(y_valpred < 86) & (y_valpred > 83)] + 2.*fac
            y_valpred[(y_valpred < 96.8) & (y_valpred > 95)] = y_valpred[(y_valpred < 96.8) & (y_valpred > 95)] - 1.*fac
            y_valpred[(y_valpred < 99) & (y_valpred > 96.8)] = y_valpred[(y_valpred < 99) & (y_valpred > 96.8)] + 1.*fac
            y_valpred[(y_valpred < 106.15) & (y_valpred > 105.2)] = y_valpred[(y_valpred < 106.15) & (y_valpred > 105.2)] - 1.*fac
            y_valpred[(y_valpred < 107) & (y_valpred > 106.15)] = y_valpred[(y_valpred < 107) & (y_valpred > 106.15)] + 1.*fac
            y_valpred[(y_valpred < 180) & (y_valpred > 120)] = y_valpred[(y_valpred < 180) & (y_valpred > 120)] - 3.*fac"""
            """y_valpred[(y_valpred < 72) & (y_valpred > 30)] = y_valpred[(y_valpred < 72) & (y_valpred > 30)] - 3.*fac
            y_valpred[(y_valpred < 83) & (y_valpred > 80)] = y_valpred[(y_valpred < 83) & (y_valpred > 80)] + 1.*fac
            y_valpred[(y_valpred < 86) & (y_valpred > 83)] = y_valpred[(y_valpred < 86) & (y_valpred > 83)] - 1.*fac
            y_valpred[(y_valpred < 96.8) & (y_valpred > 95)] = y_valpred[(y_valpred < 96.8) & (y_valpred > 95)] + 0.5*fac
            y_valpred[(y_valpred < 99) & (y_valpred > 96.8)] = y_valpred[(y_valpred < 99) & (y_valpred > 96.8)] - 0.5*fac
            y_valpred[(y_valpred < 106.15) & (y_valpred > 105.2)] = y_valpred[(y_valpred < 106.15) & (y_valpred > 105.2)] + 0.5*fac
            y_valpred[(y_valpred < 107) & (y_valpred > 106.15)] = y_valpred[(y_valpred < 107) & (y_valpred > 106.15)] - 0.5*fac
            y_valpred[(y_valpred < 180) & (y_valpred > 120)] = y_valpred[(y_valpred < 180) & (y_valpred > 120)] + 3.*fac"""

            ### Append preds and tests
            y_trainpred.extend(y_valpred)
            y_traintest.extend(y_valtest)

            r2_score = metrics.r2_score(y_valtest, y_valpred)
            if verbose >= 1: print(" (R2-score: %.04f)"%(metrics.r2_score(y_valtest, y_valpred)))

            print(y_valpred[id_valtest == 733])
            print(y_valtest[id_valtest == 733])
            print(y_valpred[id_valtest == 1957])
            print(y_valtest[id_valtest == 1957])
            print(Xc_valtest.apply(lambda x: label_dict[x.name].inverse_transform(x))[id_valtest == 1957])
            print(Xc_valtest.apply(lambda x: label_dict[x.name].inverse_transform(x))[id_valtest == 733])

            for idtr in id_valtrain:
                outlier_train[idtr].append(r2_score)
            for idte in id_valtest:
                outlier_test[idte].append(r2_score)

            if leaderboard:
                break

    ##### Compute R2-score
    r2_score = metrics.r2_score(y_traintest, y_trainpred)
    print("FINAL CV R2: ", r2_score)

    sorted_outlier_train = sorted([(key,np.mean(val)) for key,val in outlier_train.items()], key=itemgetter(1))
    sorted_outlier_test = sorted([(key,np.mean(val)) for key,val in outlier_test.items()], key=itemgetter(1))
    print(sorted_outlier_train[:10])
    print(sorted_outlier_test[:10])

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
