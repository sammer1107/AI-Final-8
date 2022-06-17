#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import scipy.stats as stats
import shap
import matplotlib.pyplot as plt
import os
from datetime import date, timedelta
from csv import *
from pycaret.regression import *


def preprocess(df, country, shift):
    df_country = df[df['CountryName'] == country]
    df_country_shift = df_country.copy()
    df_country_shift['Date'] = df_country_shift['Date'].map(lambda t: t.date()+timedelta(shift))
    df_country_shift = df_country_shift.set_index('Date')
    X = df_country_shift.iloc[:, 4:42]
    y = df_country.set_index('Date')['R_mean']
    df_merge = pd.merge(X, y, left_index=True, right_index=True)
    return df_merge


def train(df, label):
    s = setup(data=df, target=label, normalize=True, silent=True, session_id=123)
    best = compare_models(include=['dt', 'rf', 'et'])
    regression_results = pull()
    model_name = regression_results.index[0]
    best_r2 = regression_results['R2'][0]
    return best, model_name, best_r2, regression_results


def finalize(df, label, model_, country):
    s = setup(data=df, target=label, normalize=True, silent=True, session_id=123)
    model = create_model(model_)
    tuned_model = tune_model(model)
    final_model = finalize_model(tuned_model)

    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    model_path = 'saved_models/finalized_' + model_ + '_' + country
    save_model(final_model, model_path)

    return tuned_model


def cal_corr(df_shap, df, country):
    # Reference: https://github.com/dataman-git/codes_for_articles/blob/master/Explain%20your%20model%20with%20the%20SHAP%20values%20for%20article.ipynb

    shap_v = pd.DataFrame(df_shap)
    feature_list = df.columns
    shap_v.columns = feature_list
    df_v = df.copy().reset_index()

    # Determine the correlation in order to plot with different colors
    corr_list = list()
    for i in feature_list:
        b = np.corrcoef(shap_v[i], df_v[i])[1][0]
        corr_list.append(b)

    corr_df = pd.concat([pd.Series(feature_list), pd.Series(corr_list)], axis=1).fillna(0)
    # Make a data frame. Column 1 is the feature, and Column 2 is the correlation coefficient
    corr_df.columns = ['Variable', 'Corr']
    corr_df['Sign'] = np.where(corr_df['Corr'] > 0, 'red', 'blue')

    # Plot it
    shap_abs = np.abs(shap_v)
    shap_mean = pd.DataFrame(shap_v.mean()).reset_index()

    abs = pd.DataFrame(shap_abs.mean()).reset_index()
    abs.columns = ['Variable', 'SHAP_abs']
    shap_mean.columns = ['Variable', 'shap_mean']

    k2 = abs.merge(corr_df, left_on='Variable', right_on='Variable', how='inner')
    k2 = k2.sort_values(by='SHAP_abs', ascending=True)
    colorlist = k2['Sign']
    ax = k2.plot.barh(x='Variable', y='SHAP_abs', color=colorlist, legend=False, figsize=(20, 10))  # , figsize=(25,10)
    ax.set_xlabel("mean(|SHAP Value|) (Red = Positive Correlation, Blue = Negative Correlation)")
    for bars in ax.containers:
        ax.bar_label(bars)

    path = 'all_figures/' + country + '.png'
    ax.get_figure().savefig(path, bbox_inches='tight', dpi=200)
    return corr_list, list(abs['SHAP_abs']), list(shap_mean['shap_mean'])


def generate_plot(df, country, model_):
    # model_path = 'saved_models/finalized_'+model_+'_'+country
    # loaded_model = load_model(model_path)

    # explainer = shap.TreeExplainer(loaded_model.named_steps["trained_model"])
    explainer = shap.TreeExplainer(model_)
    X_orig = df.drop(["R_mean"], axis=1)
    X_normalized = X_orig.apply(stats.zscore)
    shap_values = explainer.shap_values(X_normalized)

    shap.summary_plot(shap_values, X_normalized, show=False)
    
    path = 'all_figures/' + country + '_summary.png'
    plt.savefig(path, bbox_inches='tight', dpi=200)
    plt.close()

    shap_corr, shap_abs_mean, shap_values_mean = cal_corr(shap_values, X_normalized, country)
    return shap_corr, shap_abs_mean, shap_values_mean


def insert_csv(data, opt):
    if opt == 'corr':
        fpath = 'shap_statistics/shap_correlations.csv'
        with open(fpath, 'a+', newline='') as csv_obj:
            csv_writer = writer(csv_obj)
            csv_writer.writerow(data)
    elif opt == 'shap_abs':
        fpath = 'shap_statistics/shap_abs_values.csv'
        with open(fpath, 'a+', newline='') as csv_obj:
            csv_writer = writer(csv_obj)
            csv_writer.writerow(data)
    elif opt == 'shap_mean':
        fpath = 'shap_statistics/shap_values.csv'
        with open(fpath, 'a+', newline='') as csv_obj:
            csv_writer = writer(csv_obj)
            csv_writer.writerow(data)


if __name__ == '__main__':
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    if not os.path.exists('all_figures'):
        os.makedirs('all_figures')
    if not os.path.exists('shap_statistics'):
        os.makedirs('shap_statistics')
    
    estims = pd.read_csv("data/estims_all.csv", parse_dates=['Date'])
    
    feature_list = list(estims.iloc[:, 4:42].columns)
    column = list()
    column.append('Country')
    column.append('R2')
    column.append('shift')
    column.extend(feature_list)
    fpath_shap_corr = 'shap_statistics/shap_correlations.csv'
    fpath_shap_abs = 'shap_statistics/shap_abs_values.csv'
    fpath_shap_values = 'shap_statistics/shap_values.csv'
    with open(fpath_shap_corr, 'a+', newline='') as csv_obj:
        csv_writer = writer(csv_obj)
        csv_writer.writerow(column)
    with open(fpath_shap_abs, 'a+', newline='') as csv_obj:
        csv_writer = writer(csv_obj)
        csv_writer.writerow(column)
    with open(fpath_shap_values, 'a+', newline='') as csv_obj:
        csv_writer = writer(csv_obj)
        csv_writer.writerow(column)

    search_shift = range(20)

    for country_name in set(estims['CountryName']):
        current_best_r2 = 0
        current_best_results = dict()
        current_best_results['r2'] = -100
        local_info = [country_name]

        for s in search_shift:
            df_ = preprocess(estims, country_name, s)
            model, model_name, r2, results = train(df_, 'R_mean')
            if r2 > current_best_results['r2']:
                current_best_results['model'] = model
                current_best_results['model_name'] = model_name
                current_best_results['results'] = results
                current_best_results['r2'] = r2
                current_best_results['shift'] = s

        df_final = preprocess(estims, country_name, current_best_results['shift'])
        tuned_model = finalize(df_final, 'R_mean', current_best_results['model_name'], country_name)
        shap_corr, abs_mean, shap_mean = generate_plot(df_final, country_name, tuned_model)

        local_info.append(current_best_results['r2'])
        local_info.append(current_best_results['shift'])

        local_corr = local_info.copy()
        local_corr.extend(shap_corr)
        local_abs = local_info.copy()
        local_abs.extend(abs_mean)
        local_mean = local_info.copy()
        local_mean.extend(shap_mean)

        insert_csv(local_corr, 'corr')
        insert_csv(local_abs, 'shap_abs')
        insert_csv(local_mean, 'shap_mean')
        
