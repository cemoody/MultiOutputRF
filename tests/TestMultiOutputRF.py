import unittest
import numpy as np
from MultiOutputRF import MultiOutputRF


# class TestMultiOutputRF():
class TestMultiOutputRF(unittest.TestCase):
    def test_correlated_quadratic(self):
        X1 = np.linspace(0, 1, 100)
        X2 = np.linspace(0, 1, 100)
        X = np.vstack([X1, X2]).T
        Y1 = X1**2.0 + X2**2.0
        Y2 = Y1 * 2.0
        Y = np.vstack([Y1, Y2]).T

        morf = MultiOutputRF()
        pY1 = morf.fit(X, Y)
        pY2 = morf.predict(X)
        np.allclose(pY1, Y)
        np.allclose(pY2, Y)

if __name__ == "__main__":
    t = TestMultiOutputRF()
    t.test_correlated_quadratic()
