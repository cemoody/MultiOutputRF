import time
import logging
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


class MultiOutputRF(object):
    """ Thin wrapper class around the sklearn RF Regressor. Basically allows
        you to easily predict vector values into of scalars by building a model
        for every target dimension. The main options allow us to layer the
        model so that the predictions of the first set of inputs are fed into
        a second layer of predictors that have access to all of the initially
        predicted variables. This helps in the event that the outputs are
        highly correlated.

        Methods
        -------
        fit : [n_samples, n_outputs]
        Fits one RandomForestRegressor for every n_output for every layer.

        predict : [n_samples]
        Predicts one RandomForestRegressor for every n_output propagating
        outputs from one layer to the next.
    """

    def __init__(self, func_index_rows=None, func_index_cols=None,
                 func_callback=None, metric=mean_squared_error, **kwargs):
        """
        Initialize the MultiOutputRF Model.

        Parameters
        ----------
        layers : int
        The number of layers to be built. Seperate layers have the previous
        layer of output predictions available as inputs. Default is 1 layer.

        func_index_rows : func(X, i_output) : bool array of [n_samples]
        This accepts a function that has as input and array of shape
        [n_samples, n_outputs] originally passed to fit or predict and also
        is given an integer for which output dimension MultiOutputRF is
        currently trying to predict. The function should remove any rows
        that are not relevant for predicting i_output. This can happen when
        there is missing data and we'd like to skip examples for one
        output but not another output.

        func_index_cols : func(X, i_output) : bool array of [n_outputs]
        This accepts a function that has as inputs two arrays of shape
        [n_samples, n_outputs] originally passed to fit or predict and also
        is given an integer for which output dimension MultiOutputRF is
        currently trying to predict. This is useful when that are leaky
        signals with respect to dimension i_output.
        Default is just to pass through the whole array.

        """
        self.layers = kwargs.pop('layers', 1)
        self.func_index_rows = func_index_rows
        self.func_index_cols = func_index_cols
        if func_index_rows is None:
            self.func_index_rows = lambda X, Y, i: np.ones(X.shape[0],
                                                           dtype='bool')
        if func_index_cols is None:
            self.func_index_cols = lambda X, Y, i: np.ones(X.shape[1],
                                                           dtype='bool')
        if func_callback is None:
            self.func_callback = lambda *args: None
        self.kwargs = kwargs
        self.models = {i: {} for i in range(self.layers)}
        self.logger = logging.getLogger(__name__)
        self.metric = metric

    def fit(self, X, Y):
        """
        Train the MultiOutputRF model to the data.

        Parameters
        ----------
        X : array of [n_samples, n_dimensions]
        The training input samples.

        Y : array of [n_samples, n_outputs]
        The target values, real numbers in regression.
        """
        assert X.shape[0] == Y.shape[0]
        targets = Y.columns
        self.targets = targets
        for layer in range(self.layers):
            signals_added = {}
            if len(signals_added) > 0:
                for k, v in signals_added.iteritems():
                    X[k] = v
            for target in targets:
                t0 = time.time()
                idx_rows = self.func_index_rows(X, Y, target)
                idx_cols = self.func_index_cols(X, Y, target)
                # Truncate input array rows (remove bad examples for
                # target) and cols (remove leaky signals for
                # target)
                # Training input
                tX = X.ix[idx_rows][idx_cols]
                # Scoring input
                sX = X[idx_cols]
                # Target input for subselected rows, but just for the
                # target dimension
                tY = Y.ix[idx_rows][target]
                model = RandomForestRegressor(**self.kwargs)
                assert np.prod(tX.shape) > 0
                assert np.prod(tY.shape) > 0
                model.fit(tX, tY)
                tYp = model.predict(tX)
                # Predict values for all examples
                sY = model.predict(sX)
                self.models[layer][target] = model
                self._log(layer, target, t0, tX, tY, tYp, model)
                signals_added['predicted_L%02i_%s' % (layer, target)] = sY
                self.func_callback(tX, tY)
        return np.vstack([signals_added]).T

    def _log(self, layer, target, t0, tX, tY, tYp, model):
        t1 = time.time()
        score = self.metric(tY, tYp)
        msg = 'Layer %02i, target %s, rows %1.1e, columns %1.1i, '
        msg += 'score %1.1e, training time %1.1isec'
        msg = msg % (layer, target, tX.shape[0], tX.shape[1],
                     score, t1 - t0)
        self.logger.info(msg)
        features_ranked = np.argsort(model.feature_importances_)[::-1]
        for j, ci in enumerate(features_ranked):
            v = model.feature_importances_[ci]
            n = tX.columns[ci]
            if n is not None and j < 40:
                msg = '#%02i Feature for %s: %1.2e %s'
                msg = msg % (j, target, v, n)
                self.logger.info(msg)

    def predict(self, X):
        """
        Predict targets for the MultiOutputRF model to the data.

        Parameters
        ----------
        X : array of [n_samples, n_dimensions]
        The sample data to predict targets on.
        """
        targets = self.targets
        for layer in range(self.layers):
            signals_added = {}
            if len(signals_added) > 0:
                for k, v in signals_added.iteritems():
                    X[k] = v
            for target in targets:
                t0 = time.time()
                idx_cols = self.func_index_cols(X, X, target)
                sX = X[idx_cols]
                model = self.models[layer][target]
                sY = model.predict(sX)
                # Predict values for all examples
                signals_added['predicted_L%02i_%s' % (layer, target)] = sY
                t1 = time.time()
                msg = 'Layer %02i, col %s, prediction time %1.1isec'
                msg = msg % (layer, target, t1 - t0)
                self.logger.info(msg)
        return np.vstack([signals_added]).T
