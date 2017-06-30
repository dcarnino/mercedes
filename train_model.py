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
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import ExtraTreesRegressor
#==============================================
#                   Files
#==============================================


#==============================================
#                   Functions
#==============================================
def main(verbose=1):
    """
    Main function.
    """

    ### Import data
    if verbose >= 1: print("Import data...")
    # train
    train_csv_name = "../data/mercedes/train.csv"
    df_train = pd.read_csv(train_csv_name)
    id_train = df_train["ID"].values
    y_train = df_train["y"].values
    Xb_train = df_train.iloc[:,10:].values
    Xc_train = df_train.iloc[:,2:10]
    # test
    test_csv_name = "../data/mercedes/test.csv"
    df_test = pd.read_csv(test_csv_name)
    id_test = df_test["ID"].values
    Xb_test = df_test.iloc[:,9:].values
    Xc_test = df_test.iloc[:,1:9]
    # string to numerical
    label_dict = defaultdict(LabelEncoder)
    pd.concat([Xc_train,Xc_test]).apply(lambda x: label_dict[x.name].fit(x))
    Xc_train = Xc_train.apply(lambda x: label_dict[x.name].transform(x)).values
    Xc_test = Xc_test.apply(lambda x: label_dict[x.name].transform(x)).values
    if verbose >= 3:
        print("\tid_train shape: ", id_train.shape)
        print("\ty_train shape: ", y_train.shape)
        print("\tXc_train shape: ", Xc_train.shape)
        print("\tXb_train shape: ", Xb_train.shape)
        print("\tid_test shape: ", id_test.shape)
        print("\tXc_test shape: ", Xc_test.shape)
        print("\tXb_test shape: ", Xb_test.shape)

    ### Extract features
    if verbose >= 1: print("Extract features...")
    # encode categorical
    ohe = OneHotEncoder(handle_unknown='ignore')
    ohe.fit(Xc_train)
    Xohe_train = ohe.transform(Xc_train)
    Xohe_test = ohe.transform(Xc_test)
    # merge all binary features
    print(Xohe.shape, Xb_train.shape)
    X_train = np.hstack([Xohe_train, Xb_train])
    X_test = np.hstack([Xohe_test, Xb_test])
    # remove constant
    vt = VarianceThreshold()
    vt.fit(X_train)
    X_train = vt.transform(X_train)
    X_test = vt.transform(X_test)
    if verbose >= 3:
        print("\tX_train shape: ", X_train.shape)
        print("\tX_test shape: ", X_test.shape)

    ### Train model
    if verbose >= 1: print("Train model...")
    reg = ExtraTreesRegressor(n_estimators=1000, n_jobs=-1)
    reg.fit(X_train, y_train)

    ### Predict with model
    if verbose >= 1: print("Predict with model...")
    y_pred = reg.predict(X_test)

    ### Save predictions
    if verbose >= 1: print("Save predictions...")
    pred_csv_name = "../data/mercedes/pred.csv"
    pred_df = pd.DataFrame([id_test, y_pred], columns=["ID", "y"])
    pred_df.to_csv(pred_csv_name, index=False)







#==============================================
#                   Main
#==============================================
if __name__ == '__main__':
    main(3)
