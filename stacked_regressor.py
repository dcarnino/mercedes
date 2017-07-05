from xgboost import XGBRegressor
from sklearn import metrics, model_selection
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import numpy as np



class stacked_regressor(BaseEstimator, RegressorMixin):


    def __init__(self, layer0_func, layer1_func, layer2_func, remove_bad0=0.2, remove_bad1=0.1,
                 n_folds0=5, n_folds1=5, n_est0=448, n_est1=11200, score_func=metrics.r2_score,
                 default_y_value=0.5, n_jobs=1):

        self.layer0_func = layer0_func
        self.layer1_func = layer1_func
        self.layer2_func = layer2_func

        self.remove_bad0 = remove_bad0
        self.remove_bad1 = remove_bad1

        self.n_folds0 = n_folds0
        self.n_folds1 = n_folds1

        self.n_est0 = n_est0
        self.n_est1 = n_est1

        self.score_func = score_func

        self.default_y_value = default_y_value

        self.n_jobs = n_jobs



    def fit(self, X0, y, X1=None, X2=None, verbose=0):

        X0, y = check_X_y(X0, y)
        if X1 is not None:
            X1, y = check_X_y(X1, y)
        if X2 is not None:
            X2, y = check_X_y(X2, y)

        ########## LAYER 0 ##########

        ### Init cross-validation K-folds
        cv0 = model_selection.KFold(n_splits=self.n_folds0, shuffle=True)

        ### Train first layer of regressors
        if verbose >= 1: print("Training layer 0...")
        fold_cnt = 0
        X0_0, X1_0, y_0 = [], [], []
        self.reg0_superlist_, self.score0_superlist_ = [], []
        for valtrain_index, valtest_index in cv0.split(X0):
            fold_cnt += 1
            if verbose >= 1: print("Processing fold number %d/%d..."%(fold_cnt,self.n_folds0))
            # split features and target labels
            X0_valtrain, X0_valtest = X0[valtrain_index], X0[valtest_index]
            if X1 is not None: X1_valtrain, X1_valtest = X1[valtrain_index], X1[valtest_index]
            if X2 is not None: X2_valtrain, X2_valtest = X2[valtrain_index], X2[valtest_index]
            y_valtrain, y_valtest = y[valtrain_index], y[valtest_index]
            # fit classifiers
            reg_list = self.layer0_func(input_dim=X0_valtrain.shape[1], n_jobs=self.n_jobs, n_est=self.n_est0, verbose=verbose)
            self.reg0_superlist_.append(reg_list)
            score_list = []
            X1_valpred = []
            for reg in reg_list:
                if verbose >= 2:
                    print("%s ... "%reg[0], end='')
                    sys.stdout.flush()
                if 'MLP' in reg[0]:
                    reg[1].ntraintest(X0_valtrain, y_valtrain)
                else:
                    reg[1].fit(X0_valtrain, y_valtrain)
                y_valpred = reg[1].predict(X0_valtest)
                y_valpred[np.isnan(y_valpred)] = self.default_y_value
                X1_valpred.append(y_valpred.reshape((-1,1)))
                score = self.score_func(y_valtest, y_valpred)
                score_list.append(score)
                if verbose >= 3: print("(%.04f) "%(score), end='')
            if verbose >= 2: print("")
            self.score0_superlist_.append(score_list)
            # append to new features
            X0_0.append(X0_valtest)
            if X1 is not None: X0_1.append(X1_valtest)
            if X2 is not None: X0_2.append(X2_valtest)
            X1_0.append(np.hstack(X1_valpred))
            y_0.extend(y_valtest)
        X0_0 = np.vstack(X0_0)
        if X1 is not None: X0_1 = np.vstack(X0_1)
        if X2 is not None: X0_2 = np.vstack(X0_2)
        X1_0 = np.vstack(X1_0)
        y_0 = np.array(y_0)

        ### Transpose shape of reg list of list
        self.reg0_superlist_ = list(zip(*self.reg0_superlist_))
        self.score0_superlist_ = list(zip(*self.score0_superlist_))

        ### Remove regs with too low mean score or too high std score
        mean_scores = [np.mean(tsl) for tsl in self.score0_superlist_]
        std_scores = [np.std(tsl) for tsl in self.score0_superlist_]
        mean_thresh = np.percentile(mean_scores, int(self.remove_bad0*100))
        std_thresh = np.percentile(std_scores, int((1-self.remove_bad0)*100))
        for ix_reg, (reg_l, mean_s, std_s) in enumerate(zip(self.reg0_superlist_, mean_scores, std_scores)):
            if mean_s > mean_thresh and std_s < std_thresh:
                self.reg0_superlist_.pop(ix_reg)
                X1_0 = np.delete(X1_0, ix_reg, axis=1)



        ########## LAYER 1 ##########

        if X1 is not None:
            X1 = np.hstack([X0_1, X1_0])

        ### Init cross-validation K-folds
        cv1 = model_selection.KFold(n_splits=self.n_folds1, shuffle=True)

        ### Train first layer of regressors
        if verbose >= 1: print("Training layer 1...")
        fold_cnt = 0
        X1_1, X2_1, y_1 = [], [], []
        self.reg1_superlist_, self.score1_superlist_ = [], []
        for valtrain_index, valtest_index in cv1.split(X1):
            fold_cnt += 1
            if verbose >= 1: print("Processing fold number %d/%d..."%(fold_cnt,self.n_folds1))
            # split features and target labels
            X1_valtrain, X1_valtest = X1[valtrain_index], X1[valtest_index]
            if X2 is not None: X2_valtrain, X2_valtest = X0_2[valtrain_index], X0_2[valtest_index]
            y_valtrain, y_valtest = y_0[valtrain_index], y_0[valtest_index]
            # fit classifiers
            reg_list = self.layer1_func(input_dim=X1_valtrain.shape[1], n_jobs=self.n_jobs, n_est=self.n_est1, verbose=verbose)
            self.reg1_superlist_.append(reg_list)
            score_list = []
            X2_valpred = []
            for reg in reg_list:
                if verbose >= 2:
                    print("%s ... "%reg[0], end='')
                    sys.stdout.flush()
                if 'MLP' in reg[0]:
                    reg[1].ntraintest(X1_valtrain, y_valtrain)
                else:
                    reg[1].fit(X1_valtrain, y_valtrain)
                y_valpred = reg[1].predict(X1_valtest)
                y_valpred[np.isnan(y_valpred)] = self.default_y_value
                X2_valpred.append(y_valpred.reshape((-1,1)))
                score = self.score_func(y_valtest, y_valpred)
                score_list.append(score)
                if verbose >= 3: print("(%.04f) "%(score), end='')
            if verbose >= 2: print("")
            self.score1_superlist_.append(score_list)
            # append to new features
            X1_1.append(X1_valtest)
            if X2 is not None: X1_2.append(X2_valtest)
            X2_1.append(np.hstack(X2_valpred))
            y_1.extend(y_valtest)
        X1_1 = np.vstack(X1_1)
        if X2 is not None: X1_2 = np.vstack(X0_2)
        X2_1 = np.vstack(X2_1)
        y_1 = np.array(y_1)

        ### Transpose shape of reg list of list
        self.reg1_superlist_ = list(zip(*self.reg1_superlist_))
        self.score1_superlist_ = list(zip(*self.score1_superlist_))

        ### Remove regs with too low mean score or too high std score
        mean_scores = [np.mean(tsl) for tsl in self.score1_superlist_]
        std_scores = [np.std(tsl) for tsl in self.score1_superlist_]
        mean_thresh = np.percentile(mean_scores, int(self.remove_bad1*100))
        std_thresh = np.percentile(std_scores, int((1-self.remove_bad1)*100))
        for ix_reg, (reg_l, mean_s, std_s) in enumerate(zip(self.reg1_superlist_, mean_scores, std_scores)):
            if mean_s > mean_thresh and std_s < std_thresh:
                self.reg1_superlist_.pop(ix_reg)
                X2_1 = np.delete(X2_1, ix_reg, axis=1)




        ########## LAYER 2 ##########

        if X2 is not None:
            X2 = np.hstack([X1_2, X2_1])

        ### Init final layer
        self.reg_final_ = self.layer2_func(n_jobs=n_jobs, verbose=verbose)

        ### Train final layer
        if verbose >= 1: print("Training layer 2...")
        self.reg_final_.fit(X2, y_2)




    def predict(self, X0, X1=None, X2=None, verbose=0):

        check_is_fitted(self, ['reg0_superlist_', 'score0_superlist_', 'reg1_superlist_', 'score1_superlist_', 'reg_final_'])

        X = check_array(X)

        ### Predict with both layers
        # layer 0
        if verbose >= 1: print("Predictions of layer 0...")
        X1_0 = []
        for reg_list in self.reg0_superlist_:
            y_subpred = [reg[1].predict(X0) for reg in reg_list]
            y_subpred = sum(y_subpred)/float(len(reg_list))
            X1_0.append(y_subpred.reshape((-1,1)))
        X1_0 = np.hstack(X1_0)
        # layer 1
        if X1 is not None:
            X1 = np.hstack([X1, X1_0])
        if verbose >= 1: print("Predictions of layer 1...")
        X2_1 = []
        for reg_list in self.reg1_superlist_:
            y_subpred = [reg[1].predict(X1) for reg in reg_list]
            y_subpred = sum(y_subpred)/float(len(reg_list))
            X2_1.append(y_subpred.reshape((-1,1)))
        X2_1 = np.hstack(X2_1)
        # layer 2
        if X2 is not None:
            X2 = np.hstack([X2, X2_1])
        if verbose >= 1: print("Predictions of layer 2...")
        y_pred = self.reg_final_.predict(X2)

        return y_pred
