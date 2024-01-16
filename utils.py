import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import sklearn
sklearn.set_config(transform_output="pandas")


class LogOddsEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, smoothing_factor=1):
        self.feature_log_odds = {}
        self.smoothing_factor = smoothing_factor

    def fit(self, X, y=None):
        for column in X.columns:
            # Calculate log odds for each category in the column
            Xy = pd.concat([X,y], axis=1)
            events = Xy.groupby(column)['target'].mean()
            category_log_odds = np.log((events + self.smoothing_factor) / 
                                       (Xy['target'].mean() + 2 * self.smoothing_factor))
            
            self.feature_log_odds[column] = category_log_odds.to_dict()
            
        return self

    def transform(self, X, y=None):
        X_encoded = pd.DataFrame(X, columns=self.feature_log_odds.keys())
        for column in X_encoded.columns:
            # Replace original categories with log odds in the transformed data
            X_encoded[column] = X_encoded[column].map(self.feature_log_odds[column])
        return X_encoded
    
    def get_feature_names_out():
        pass
