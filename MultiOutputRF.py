import numpy as np
from sklearn.ensemble import RandomForestRegressor

passthrough = lambda Y, i: Y


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

        remove_leaky_signals : func(X, i_output) : X'
        This accepts a function that has as input and array of shape
        [n_samples, n_outputs] originally passed to fit or predict and also
        is given an integer for which output dimension MultiOutputRF is
        currently trying to predict. The function should remove any signals
        that are leaky with respect to dimention i_output and return the
        sanitized array. Default is just to pass through the whole array.

        """
        self.args = args
        self.layers = kwargs.pop('layers', 1)
        self.remove_leaky_signals = kwargs.pop('remove_leaky_signals',
                                               passthrough)
        self.kwargs = kwargs
        self.models = {i: {} for i in range(self.layers)}

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
        cX = X
        for layer in range(self.layers):
            signals_added = []
            if len(signals_added) > 0:
                cX = np.hstack([cX, signals_added])
            for i in range(Ny):
                iX = self.remove_leaky_signals(cX.copy(), i)
                model = RandomForestRegressor(*self.args, **self.kwargs)
                predicted_Y = model.fit_transform(iX, Y[:, i])
                self.models[layer][i] = model
                signals_added.append(predicted_Y)
        return np.vstack([signals_added]).T

    def predict(self, X):
        """
        Predict targets for the MultiOutputRF model to the data.

        Parameters
        ----------
        X : array of [n_samples, n_dimensions]
        The sample data to predict targets on.
        """
        cX = X
        for layer in range(self.layers):
            signals_added = []
            if len(signals_added) > 0:
                cX = np.hstack([cX, signals_added])
            Ny = len(self.models[layer].values())
            for i in range(Ny):
                model = self.models[layer][i]
                predicted_Y = model.predict(cX)
                self.models[layer][i] = model
                signals_added.append(predicted_Y)
        return np.vstack([signals_added]).T
