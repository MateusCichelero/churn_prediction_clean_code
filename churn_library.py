"""
Module for identification of credit card customers most likely to churn.
Implements EDA, model training and evaluation.

Author: Mateus Cichelero
Date: May 2022
"""


# import libraries
# import constants (including projects paths)
import os
from typing import Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from feature_engine.encoding import MeanEncoder
from constants import PATH_TO_ROC_CURVE
from constants import PATH_TO_LR_MODEL
from constants import PATH_TO_RFC_MODEL
from constants import PATH_TO_RF_RESULTS
from constants import PATH_TO_LR_RESULTS
from constants import PATH_TO_HEATMAP
from constants import PATH_TO_TRANSACTION_DISTRIBUTION
from constants import PATH_TO_MARITAL_DISTRIBUTION
from constants import PATH_TO_AGE_DISTRIBUTION
from constants import PATH_TO_CHURN_DISTRIBUTION
from constants import PATH_TO_FEATURE_IMPORTANCES
from constants import KEEP_COLS
from constants import CAT_COLUMNS
from constants import PATH_TO_DATA
from constants import PARAM_GRID

sns.set()
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth: str) -> pd.DataFrame:
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df_data: pandas dataframe
    '''
    df_data = pd.read_csv(pth)

    return df_data


def perform_eda(df_data: pd.DataFrame) -> None:
    '''
    perform eda on df and save figures to images folder
    input:
            df_data: pandas dataframe

    output:
            None
    '''
    plt.figure(figsize=(20, 10))

    df_data['Churn'].hist()
    plt.savefig(PATH_TO_CHURN_DISTRIBUTION)

    df_data['Customer_Age'].hist()
    plt.savefig(PATH_TO_AGE_DISTRIBUTION)

    df_data['Marital_Status'].value_counts('normalize').plot(kind='bar')
    plt.savefig(PATH_TO_MARITAL_DISTRIBUTION)

    sns.histplot(df_data['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig(PATH_TO_TRANSACTION_DISTRIBUTION)

    sns.heatmap(df_data.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(PATH_TO_HEATMAP)


def encoder_helper(
        df_data: pd.DataFrame,
        category_lst: list,
        response: str) -> pd.DataFrame:
    '''
    helper function to turn each categorical column into a new column with
    proportion of churn for each category - associated with cell 15 from the notebook

    input:
            df_data: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for
            naming variables or index y column]

    output:
            df: pandas dataframe with new columns
    '''

    # Using feature-engine library for automating categorical mean encoding:
    # https://feature-engine.readthedocs.io/en/1.3.x/api_doc/encoding/MeanEncoder.html
    # !CONCEPT WARNING!: In a more robust modeling process, the feature encoding step would be
    # applied only after train/test data splitting (fitted on train set and applied to both)
    mean_enc = MeanEncoder(variables=category_lst)
    mean_enc.fit(df_data, df_data[response])
    df_data = mean_enc.transform(df_data)

    # rename mean new encoded columns
    new_names = [(column_name, column_name + '_' + response)
                 for column_name in df_data.columns.values if column_name in category_lst]
    df_data.rename(columns=dict(new_names), inplace=True)

    return df_data


def perform_feature_engineering(
        df_data: pd.DataFrame,
        response: str = 'Churn') -> list:
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for
              naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

    df_data = encoder_helper(df_data, CAT_COLUMNS, response)
    x_original = df_data[KEEP_COLS]
    y_original = df_data[response]

    # train test split
    x_train, x_test, y_train, y_test = train_test_split(
        x_original, y_original, test_size=0.3, random_state=42)

    return x_train, x_test, y_train, y_test


def classification_report_image(y_data: list,
                                y_train_preds_lr: list,
                                y_train_preds_rf: list,
                                y_test_preds_lr: list,
                                y_test_preds_rf: list) -> None:
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    y_train, y_test = y_data

    # Generating and saving train and test classification report for lr:
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                y_train, y_train_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                y_test, y_test_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(PATH_TO_LR_RESULTS)

    # Generating and saving train and test classification report for rf:
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(PATH_TO_RF_RESULTS)


def feature_importance_plot(
        model: Any,
        x_data: pd.DataFrame,
        output_pth: str) -> None:
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth)


def train_models(
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: list,
        y_test: list) -> None:
    '''
    train, store model results: images + scores, and store models
    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''

    # define classifiers
    rfc = RandomForestClassifier(random_state=42)
    lr_model = LogisticRegression(solver='lbfgs', max_iter=3000)

    # perform grid search and fit models to data
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=PARAM_GRID, cv=5)
    cv_rfc.fit(x_train, y_train)
    lr_model.fit(x_train, y_train)

    # generate prediction data
    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lr_model.predict(x_train)
    y_test_preds_lr = lr_model.predict(x_test)

    rfc_model = cv_rfc.best_estimator_

    # save models to disk
    joblib.dump(rfc_model, PATH_TO_RFC_MODEL)
    joblib.dump(lr_model, PATH_TO_LR_MODEL)

    # generate and save classification reports
    classification_report_image(
        [y_train,y_test],
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf)

    # generate and save roc curve
    lrc_plot = plot_roc_curve(lr_model, x_test, y_test)
    plt.figure(figsize=(15, 8))
    ax_plt = plt.gca()
    rfc_plot = plot_roc_curve(rfc_model, x_test, y_test, ax=ax_plt, alpha=0.8)
    rfc_plot.plot(ax=ax_plt, alpha=0.8)
    lrc_plot.plot(ax=ax_plt, alpha=0.8)
    plt.savefig(PATH_TO_ROC_CURVE)

    # generate and save feature importances
    feature_importance_plot(rfc_model, x_test, PATH_TO_FEATURE_IMPORTANCES)


if __name__ == '__main__':

    # import data
    df_churn = import_data(PATH_TO_DATA)
    # add Churn binary target column
    df_churn['Churn'] = df_churn['Attrition_Flag'].apply(
        lambda val: 0 if val == 'Existing Customer' else 1)
    # perform exploratory data analysis, saving results as images
    perform_eda(df_churn)
    # perform feature eng and train/test splitting of data
    x_tr, x_tst, y_tr, y_tst = perform_feature_engineering(
        df_churn, 'Churn')
    # train and persist logist regression and random forest classification
    # models, saving results as images
    train_models(x_tr, x_tst, y_tr, y_tst)
