import logging as log
import numpy as np
import os

from sklearn.model_selection import train_test_split

from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback


class KerasRegressor(KerasRegressor):
    """
    Add ntraintest to traintest until stopping criteria met.
    """

    def ntraintest(self, x, y, scoresign=-1, **params):
        """ fit until stopping criteria met and retain best model """
        xtrain, xtest, ytrain, ytest = train_test_split(x, y,
                                        test_size=0.1, random_state=0)
        params.setdefault("verbose", 0)
        # keras verbose is 0 or 1 only
        if params["verbose"] > 0:
            params["verbose"] = 1

        earlystopping = EarlyStopping(monitor='val_loss',
                                patience=50, verbose=0, mode='auto')
        best_file = './best_weights.txt'
        savebestmodel = ModelCheckpoint(best_file,
                                monitor='val_loss', verbose=0,
                                save_best_only=True, mode='auto')
        keraslog = Keraslog()
        self.fit(xtrain, ytrain,
                validation_data=(xtest, ytest),
                callbacks=[earlystopping, savebestmodel, keraslog],
                **params)
        self.model.load_weights(best_file)
        os.remove(best_file)


        losses = np.array(keraslog.losses)

        losses = losses * scoresign
        self.best_score = max(losses)
        best_iteration = list(losses).index(self.best_score) + 1

        log.info("best score %s with %s iterations"
                         %(self.best_score, best_iteration))

        return losses


class Keraslog(Callback):
    """ log scores from keras iterations """
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('val_loss'))

    def on_train_end(self, logs={}):
        logs.clear()
