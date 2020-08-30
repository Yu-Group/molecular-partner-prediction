from .forest import RandomForestClassifier
class wrf(RandomForestClassifier):
    def fit(self, X, y, sample_weight=None, feature_weight=None, K = 5):
        for k in range(K):
            if k == 0:
                # Initially feature weights are None
                feature_importances = feature_weight
                
                # fit the classifier
                super(wrf, self).fit(X=X,
                         y=y,
                         feature_weight=feature_importances)
                
                # Update feature weights using the
                # new feature importance score
                feature_importances = self.feature_importances_

            else:
                # fit weighted RF
                # fit the classifier
                super(wrf, self).fit(
                        X=X,
                        y=y,
                        feature_weight=feature_importances)
                
                # Update feature weights using the
                # new feature importance score
                feature_importances = self.feature_importances_
        return self

#Eric: Doesn't lfook like much is changed here, as it looks like no significant differences between regressor and classifier
#       come back to later to make sure there really is no difference                
from .forest import RandomForestRegressor # Hue: need to import the class regressor first
class wrf_reg(RandomForestRegressor): # Hue: change the name so that it does not clash with the first one.
    def fit(self, X, y, sample_weight=None, feature_weight=None, K = 5):
        for k in range(K):
            if k == 0:
                # Initially feature weights are None
                feature_importances = feature_weight
                
                # fit the classifier
                super(wrf_reg, self).fit(X=X,
                         y=y,
                         feature_weight=feature_importances)
                
                # Update feature weights using the
                # new feature importance score
                feature_importances = self.feature_importances_

            else:
                # fit weighted RF
                # fit the regressor
                super(wrf_reg, self).fit(
                        X=X,
                        y=y,
                        feature_weight=feature_importances)
                
                # Update feature weights using the
                # new feature importance score
                feature_importances = self.feature_importances_
        return self
