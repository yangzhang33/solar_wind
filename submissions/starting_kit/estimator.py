from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA 
import xgboost as xgb
import numpy as np
import pandas as pd



def compute_rolling_std(data, feature, time_window, center=False):
    """
    For a given dataframe, compute the standard deviation over
    a defined period of time (time_window) of a defined feature
    Parameters
    ----------
    data : dataframe
    feature : str
        feature in the dataframe we wish to compute the rolling mean from
    time_indow : str
        string that defines the length of the time window passed to `rolling`
    center : bool
        boolean to indicate if the point of the dataframe considered is
        center or end of the window
    """
    name = '_'.join([feature, time_window, 'std'])
    data[name] = data[feature].rolling(time_window, center=center).std()
    data[name] = data[name].ffill().bfill()
    data[name] = data[name].astype(data[feature].dtype)
    return data

def compute_rolling_mean(data, feature, time_window, center=False):
    
    name = '_'.join([feature, time_window, 'mean'])
    data[name] = data[feature].rolling(time_window, center=center).mean()
    data[name] = data[name].ffill().bfill()
    data[name] = data[name].astype(data[feature].dtype)
    return data

def compute_rolling_median(data, feature, time_window, center=False):
    
    name = '_'.join([feature, time_window, 'median'])
    data[name] = data[feature].rolling(time_window, center=center).median()
    data[name] = data[name].ffill().bfill()
    data[name] = data[name].astype(data[feature].dtype)
    return data

class FeatureExtractor(object):

    def fit(self, X_df, y):
        return self

    def transform(self, X_df):
        X_df_new = X_df.copy()
        X_df_new=X_df_new.drop(['Range F 0'], axis=1)
        X_df_new=X_df_new.drop(['Range F 2'], axis=1)
        X_df_new=X_df_new.drop(['Range F 3'], axis=1)
        X_df_new=X_df_new.drop(['Range F 4'], axis=1)
        X_df_new=X_df_new.drop(['Range F 7'], axis=1)
        X_df_new=X_df_new.drop(['Range F 9'], axis=1)
        X_df_new=X_df_new.drop(['Range F 10'], axis=1)
        X_df_new=X_df_new.drop(['Range F 12'], axis=1)
        X_df_new=X_df_new.drop(['Range F 13'], axis=1)
        X_df_new=X_df_new.drop(['Range F 14'], axis=1)

        features = X_df_new.columns
        for i in ['2h', '12h'] :
            for j in features: 
                X_df_new = compute_rolling_std(X_df_new, j, i)
                X_df_new = compute_rolling_mean(X_df_new, j, i)
                #X_df_new = compute_rolling_median(X_df_new, j, i)
        for i in features:
            X_df_new[i+'_mean_delta_2h']=X_df_new[i] - X_df_new[i].shift(24)
        for i in features:
            X_df_new = X_df_new.drop([i], axis=1)
        X_df_new = X_df_new.fillna(0)    
        return X_df_new
        
class Classifier(BaseEstimator):
    def __init__(self):

        xgboost_model = xgb.XGBRegressor(booster = 'gbtree',objective = 'binary:logistic', colsample_bytree = 0.9, learning_rate = 0.1,
                max_depth = 7, alpha = 10, n_estimators = 50 , eval_metric = 'auc')
        
        self.model = xgboost_model

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        pred = self.model.predict(X)
        y_pred = np.asarray([[1-i, i] for i in pred])
        y_pred_smooth = pd.DataFrame(data = y_pred).rolling(12, min_periods =0, center=True).quantile(0.90).values
        return y_pred_smooth


def get_estimator():
    feature_extractor = FeatureExtractor()
    classifier = Classifier()

    pipe = make_pipeline(feature_extractor, StandardScaler(), classifier)

    return pipe
