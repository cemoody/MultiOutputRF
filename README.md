MultiOutputRF
=============

This class (thinly) extends Scikit-Learn Random Forests for Vectors

The standard sklearn RFs allow for regressing multiple outputs at a time in a simplistic way. When splitting a scalar at the leaf level, the RFs try to maximize the resulting impurity. When you've got to split on a vector, this gets trickier and the sklearn implementation just takes the average impurity over all classes. This is likely not as performant as building a separate model for each dimension in the output -- which is what this module helps you do.

Additionally, this class has callbacks that allow you to remove irrelevant rows and columns. You might like to remove rows that are good example observations for one target variable, but are bad, missing, or irrelevant for another target variable. You might like to remove columns dynamically for certain features: for a leaky signal for target A could be a great correlated signal for target B.

Example
=======

Usage is exactly the same as sklearn's typical RF implementation, and is just a thin wrapper around it for running multiple models. Excepting the options MultiObjectRF itself uses, any kwargs passed to the MultiObjectRF will be passed down to each individual RandomForest module.

    from MultiOutputRF import MultiOutputRF
    import numpy as np

    X1 = np.linspace(0, 1, 100)
    X2 = np.linspace(0, 1, 100)
    X = np.vstack([X1, X2]).T
    Y1 = X1**2.0 + X2**2.0
    Y2 = Y1 * 2.0
    Y = np.vstack([Y1, Y2]).T

    morf = MultiOutputRF()
    pY1 = morf.fit(X, Y)
    pY2 = morf.predict(X)

Parameters 
==========

    func_index_rows : func(X, Y, i_output) : bool array of [n_samples]

This accepts a function that has as input and array of shape
[n_samples, n_outputs] originally passed to fit or predict and also
is given an integer for which output dimension MultiOutputRF is
currently trying to predict. The function should remove any rows
that are not relevant for predicting i_output. This can happen when
there is missing data and we'd like to skip examples for one
output but not another output.

    func_index_cols : func(X, Y, i_output) : bool array of [n_outputs]

This accepts a function that has as inputs two arrays of shape
[n_samples, n_outputs] originally passed to fit or predict and also
is given an integer for which output dimension MultiOutputRF is
currently trying to predict. This is useful when that are leaky
signals with respect to dimension i_output.
Default is just to pass through the whole array.
