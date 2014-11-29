import numpy as np
import logging
from sklearn.ensemble import RandomForestRegressor


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

    def __init__(self, *args, **kwargs):
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
        self.args = args
        self.layers = kwargs.pop('layers', 1)
        passthrough_rows = lambda X, Y, i: np.ones(X.shape[0], dtype='bool')
        passthrough_cols = lambda X, Y, i: np.ones(X.shape[1], dtype='bool')
        self.func_index_rows = kwargs.pop('func_index_rows', passthrough_rows)
        self.func_index_cols = kwargs.pop('func_index_cols', passthrough_cols)
        self.kwargs = kwargs
        self.models = {i: {} for i in range(self.layers)}
        self.logger = logging.getLogger(__name__)

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
        X, Y = map(np.atleast_2d, (X, Y))
        assert X.shape[0] == Y.shape[0]
        Ny = Y.shape[1]
        for layer in range(self.layers):
            signals_added = []
            if len(signals_added) > 0:
                X = np.hstack([X, signals_added])
            for i in range(Ny):
                idx_rows = self.func_index_rows(X, Y, i)
                idx_cols = self.func_index_cols(X, Y, i)
                # Truncate input array rows (remove bad examples for
                # target i) and cols (remove leaky signals for
                # target i)
                tX = X[idx_rows, :][:, idx_cols]
                # Target array for subselected rows, but just for the
                # target dimension
                tY = Y[idx_rows, i]
                msg = 'Layer %02i, target %02i, rows %1.1e, columns %1.1i '
                msg = msg % (layer, i, idx_rows.sum(), idx_cols.sum())
                self.logger.info(msg)
                model = RandomForestRegressor(*self.args, **self.kwargs)
                assert tX.size > 0
                assert tY.size > 0
                model.fit(tX, tY)
                # Predict values for all examples
                fX = X[:, idx_cols]
                fY = model.predict(fX)
                self.models[layer][i] = model
                signals_added.append(fY)
        return np.vstack([signals_added]).T

    def predict(self, X):
        """
        Predict targets for the MultiOutputRF model to the data.

        Parameters
        ----------
        X : array of [n_samples, n_dimensions]
        The sample data to predict targets on.
        """
        X = np.atleast_2d(X)
        for layer in range(self.layers):
            signals_added = []
            if len(signals_added) > 0:
                X = np.hstack([X, signals_added])
            Ny = len(self.models[layer].values())
            for i in range(Ny):
                model = self.models[layer][i]
                idx_cols = self.func_index_cols(X, X, i)
                # Predict values for all examples
                fX = X[:, idx_cols]
                fY = model.predict(fX)
                signals_added.append(fY)
        return np.vstack([signals_added]).T
