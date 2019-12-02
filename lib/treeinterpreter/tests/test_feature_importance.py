#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_feature_importance
----------------------------------

Tests for `feature_importance` module.
"""

import numpy as np
import unittest

from sklearn.datasets import load_boston, load_iris
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                              ExtraTreesClassifier, ExtraTreesRegressor,)
from sklearn.preprocessing import OneHotEncoder

from treeinterpreter import treeinterpreter, feature_importance
from treeinterpreter.feature_importance import feature_importance
class TestFeatureImportance(unittest.TestCase):

    def setUp(self):
        self.boston = load_boston()
        self.iris = load_iris()

    def test_forest_regressor(self):
        for ForestRegressor in (RandomForestRegressor, ExtraTreesRegressor):
            X = self.boston.data
            Y = self.boston.target
            
            #Predict for decision tree
            dt = ForestRegressor(n_estimators=10)
            dt.fit(X, Y)
            Y_2d = Y[:, np.newaxis]
            out, _ = feature_importance(dt, X, Y_2d, type = 'classic')
            self.assertTrue(np.allclose(dt.feature_importances_, out))
        
    def test_forest_classifier(self):
        for ForestClassifier in (RandomForestClassifier, ExtraTreesClassifier):
            X = self.iris.data
            Y = self.iris.target
            dt = ForestClassifier(max_depth=3)
            dt.fit(X, Y)
            Y_2d = OneHotEncoder().fit_transform(Y[:,np.newaxis]).todense()
            out, _ = feature_importance(dt, X, Y_2d, type = 'classic')
            self.assertTrue(np.allclose(dt.feature_importances_, out))

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
