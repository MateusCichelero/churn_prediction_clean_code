"""
Module for testing and logging of the churn_library project

Author: Mateus Cichelero
Date: May 2022
"""

import logging
from os.path import exists
import pytest
import churn_library as clib
import constants

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import():
    '''
    test data import - checks if data is imported correctly and is not empty.
    '''
    try:
        df_data = clib.import_data(constants.PATH_TO_DATA)
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

    try:
        assert df_data.shape[0] > 0
        assert df_data.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err

    # persisting dataframe in Namespace.
    # check conftest.py file.
    pytest.df = df_data


def test_eda():
    '''
    test perform eda - check function and saved images
    '''

    df_data = pytest.df
    df_data['Churn'] = df_data['Attrition_Flag'].apply(
        lambda val: 0 if val == 'Existing Customer' else 1)

    try:
        clib.perform_eda(df_data)
        logging.info("Testing perform_eda function: SUCCESS")
    except KeyError as err:
        logging.error('Missing columns "%s" in the dataset', err.args[0])
        raise err

    try:
        assert exists(constants.PATH_TO_CHURN_DISTRIBUTION) is True
        assert exists(constants.PATH_TO_AGE_DISTRIBUTION) is True
        assert exists(constants.PATH_TO_MARITAL_DISTRIBUTION) is True
        assert exists(constants.PATH_TO_TRANSACTION_DISTRIBUTION) is True
        assert exists(constants.PATH_TO_HEATMAP) is True
        logging.info("Testing perform_eda saving files: SUCCESS")
    except AssertionError as err:
        logging.error("Testing perform_eda: Resulting image files not in directory")
        raise err


def test_encoder_helper():
    '''
    test encoder helper - checks if all expected encoded columns are generated
    '''
    df_data = pytest.df

    try:
        df_data = clib.encoder_helper(df_data, constants.CAT_COLUMNS, "Churn")
        assert set(constants.KEEP_COLS).issubset(set(df_data.columns))
        logging.info("Testing encoder_helper function: SUCCESS")
    except AssertionError as err:
        logging.error('Testing encoder_helper: encoding process failed, not all columns generated')
        raise err


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''
    df_data = pytest.df

    try:
        x_train, x_test, y_train, y_test = clib.perform_feature_engineering(df_data, "Churn")
        assert x_train.shape[0] > 0
        assert x_test.shape[0] > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error("Testing perform_feature_engineering: failed to generate the splitted data")
        raise err

    pytest.x_train = x_train
    pytest.x_test = x_test
    pytest.y_train = y_train
    pytest.y_test = y_test

def test_train_models():
    '''
    test train_models
    '''
    x_train = pytest.x_train
    x_test = pytest.x_test
    y_train = pytest.y_train
    y_test = pytest.y_test

    # general check
    try:
        clib.train_models(x_train, x_test, y_train, y_test)
        logging.info("Testing train_models general function flow: SUCCESS")
    except BaseException as err:
        logging.error("Testing train_models: failed to train the models")
        raise err

    # model saving check
    try:
        assert exists(constants.PATH_TO_RFC_MODEL) is True
        assert exists(constants.PATH_TO_LR_MODEL) is True
        logging.info("Testing train_models saving models: SUCCESS")
    except AssertionError as err:
        logging.error("Testing train_models saving models: model files not saved")
        raise err

    # classification report images saving check
    try:
        assert exists(constants.PATH_TO_LR_RESULTS) is True
        assert exists(constants.PATH_TO_RF_RESULTS) is True
        logging.info("Testing train_models saving classification reports: SUCCESS")
    except AssertionError as err:
        logging.error("Testing train_models saving classification reports: files not saved")
        raise err

    # roc curve image saving check
    try:
        assert exists(constants.PATH_TO_ROC_CURVE) is True
        logging.info("Testing train_models saving roc curve: SUCCESS")
    except AssertionError as err:
        logging.error("Testing train_models saving roc curve: file not saved")
        raise err

    # feature importances image saving check
    try:
        assert exists(constants.PATH_TO_FEATURE_IMPORTANCES) is True
        logging.info("Testing train_models saving feature importances: SUCCESS")
    except AssertionError as err:
        logging.error("Testing train_models saving feature importances: file not saved")
        raise err


if __name__ == "__main__":
    test_import()
    test_eda()
    test_encoder_helper()
    test_perform_feature_engineering()
    test_train_models()
