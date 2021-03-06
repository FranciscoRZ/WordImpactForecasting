from random import choice

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, KFold
from xgboost.sklearn import XGBClassifier
from imblearn.over_sampling import SMOTE

#region DATAEXPLORATION
def numerical_description(variables:list, data:pd.DataFrame) -> pd.DataFrame:
    '''
    Produces a numerical description of the features
    '''
    numeric_describe = data[variables].describe()
    numeric_describe.loc["count"] = numeric_describe.loc["count"].astype('int64')
    min_max = numeric_describe.loc["max"] - numeric_describe.loc["min"]
    min_max.name = "Max min diff"
    numeric_describe = numeric_describe.append(min_max)
    return numeric_describe

def __plot_correlation_matrix_plt(corr:pd.DataFrame, title:str):
    '''
    Plots the input correlation matrix with pyplot
    '''
    fig = plt.figure(figsize=(12,12))
    plt.matshow(corr, fignum = fig.number)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
    plt.yticks(range(len(corr.columns)), corr.columns)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=10)
    plt.title(title, fontsize=16, pad=30);
    return

def __plot_correlation_matrix_sns(corr: pd.DataFrame, title:str):
    '''
    Plots the input correlation matrix with seaborn
    '''
    fig, ax = plt.subplots(figsize=(12,12))
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
    plt.title(title, pad=30);
    return

def plot_correlation_matrix(corr: pd.DataFrame, title:str, which:str):
    '''
    Plots the input correlation matrix with the desired library
    '''
    if which=='pyplot':
        return __plot_correlation_matrix_plt(corr, title)
    elif which=='seaborn':
        return __plot_correlation_matrix_sns(corr, title)
    else:
        raise ValueError("Plotting method not implemented. Choose 'pyplot' or 'seaborn'")

def compute_day_count(ticker:str, data:pd.DataFrame) -> dict:
    '''
    Computes the day count of observations for given ticker
    '''
    tmp = data.loc[data['TICKER']==ticker]
    day_count = {}
    for year in data["annee"].unique():
        day_count[year] = tmp.loc[tmp["annee"]==year].set_index(["annee", "mois", "jour"]).shape[0]
    return day_count
#endregion

#region PREPROCESSING
def handle_outliers(data:pd.DataFrame, features:list, how:str) -> pd.DataFrame:
    tickers = data["TICKER"].unique()
    if how=='remove':
        all_indeces = []
        for ticker in tickers:
            tmp = data.loc[data["TICKER"]==ticker,features]
            mask = np.where(abs(stats.zscore(tmp)) > 3)
            all_indeces = all_indeces + list(mask[0])
        return data.drop(all_indeces)
    elif how=='cap':
        new_data = data.copy()
        for ticker in tickers:
            tmp = new_data.loc[data["TICKER"]==ticker, features]
            new_data.loc[tmp.index, features].clip(lower=data.loc[tmp.index, features].quantile(0.25),
                                                   upper=data.loc[tmp.index, features].quantile(0.75), 
                                                   inplace=True, axis='columns')
            
        return new_data
        
def mean_encode(X:pd.DataFrame, y:pd.Series, feature:str) -> pd.Series:
    tmp = X.merge(y, right_index=True, left_index=True)
    mean_encoder = tmp.groupby(feature).apply(np.mean)["y"].copy()
    return mean_encoder
#endregion

#region MODEL_SELECTION
def classification_metrics(type_model:str, parameters:dict,
                           y_cv:list, predictions:list) -> tuple:
    ''' Computes classification metrics for input data
    '''
    auc = round(roc_auc_score(y_cv, predictions), 4)
    precision = round(precision_score(y_cv, predictions), 4)
    recall = round(recall_score(y_cv, predictions), 4)
    return ([{'algorithm':type_model,
             'parametres': parameters,
             'metriques': {'auc':auc,
                           'precision':precision,
                           'recall':recall}
             }])

def tune_XGB_hyper_params(hyper_parameter_grid:list, X_train:pd.DataFrame,
                          y_train:pd.DataFrame, num_folds:int, num_trials:int) -> list:
    '''
    Computes the classification metrics for all the parameters sets in the
    grid
    '''
    results = []
    kf = KFold(n_splits=num_folds)
    for n in range(num_trials):
        hyper_param_set = choice(hyper_parameter_grid)

        param_learning_rate = hyper_param_set[0]
        param_max_depth = hyper_param_set[1]
        param_n_estimators = hyper_param_set[2]
        param_subsample = hyper_param_set[3]
        param_colsample_bytree = hyper_param_set[4]
        param_colsample_bylevel = hyper_param_set[5]
        param_gamma = hyper_param_set[6]
        
        y_cv = []
        predictions = []
        for train_index, valid_index in kf.split(X_train, y_train):
            xgb = XGBClassifier(booster='gbtree', objective='binary:logistic',
                                learning_rate=param_learning_rate,
                                max_depth=param_max_depth,
                                n_estimators=param_n_estimators,
                                subsample=param_subsample,
                                colsample_bytree=param_colsample_bytree,
                                colsample_bylevel=param_colsample_bylevel,
                                gamma=param_gamma)

            X_train_k, X_valid_k = X_train.iloc[train_index].copy(), X_train.iloc[valid_index].copy()
            y_train_k, y_valid_k = y_train.iloc[train_index].copy(), y_train.iloc[valid_index].copy()

            over_sample = SMOTE()
            X_train_k, y_train_k = over_sample.fit_resample(X_train, y_train)
            
            xgb_model = xgb.fit(X_train_k, y_train_k.y)
            pred_valid = xgb_model.predict(X_valid_k)
            y_cv = y_cv + y_valid_k.y.to_list()
            predictions = predictions + pred_valid.tolist()

        results = results + classification_metrics("XGBClassifier",
                                             {'learning_rate':param_learning_rate,
                                              'max_depth':param_max_depth,
                                              'n_estimators':param_n_estimators,
                                              'subsample':param_subsample,
                                              'colsample_bytree':param_colsample_bytree,
                                              'colsample_bylevel':param_colsample_bylevel,
                                              'gamma':param_gamma},
                                              y_cv, predictions)
    return results

def plot_tuning_results(results:list, param_name:str):
    '''
    Plots the results of hyper parameter tuning for required parameter
    '''
    auc = [res['metriques']['auc'] for res in results]
    precision = [res['metriques']['precision'] for res in results]
    recall = [res['metriques']['recall'] for res in results]
    param = [res['parametres'][param_name] for res in results]

    fig, ax = plt.subplots(1,3, figsize=(15,5))
    ax[0].scatter(param, auc)
    ax[0].set_xlabel(param_name)
    ax[0].set_ylabel('Validation AUC')
    ax[1].scatter(param, precision)
    ax[1].set_xlabel(param_name)
    ax[1].set_ylabel('Validation Precision')
    ax[2].scatter(param, recall)
    ax[2].set_xlabel(param_name)
    ax[2].set_ylabel('Validation recall');

    return
