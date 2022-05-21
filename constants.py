"""
Module that stores constants used in project

Author: Mateus Cichelero
Date: May 2022
"""

PATH_TO_DATA = './data/bank_data.csv'
PATH_TO_FEATURE_IMPORTANCES = './images/results/feature_importances.png'
PATH_TO_CHURN_DISTRIBUTION = './images/eda/churn_distribution.png'
PATH_TO_AGE_DISTRIBUTION = './images/eda/customer_age_distribution.png'
PATH_TO_MARITAL_DISTRIBUTION = './images/eda/marital_status_distribution.png'
PATH_TO_TRANSACTION_DISTRIBUTION = './images/eda/total_transaction_distribution.png'
PATH_TO_HEATMAP = './images/eda/heatmap.png'
PATH_TO_LR_RESULTS = './images/results/logistic_results.png'
PATH_TO_RF_RESULTS = './images/results/rf_results.png'
PATH_TO_RFC_MODEL = './models/rfc_model.pkl'
PATH_TO_LR_MODEL = './models/logistic_model.pkl'
PATH_TO_ROC_CURVE = './images/results/roc_curve_result.png'

CAT_COLUMNS = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category']

KEEP_COLS = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
             'Income_Category_Churn', 'Card_Category_Churn']

PARAM_GRID = { 
    'n_estimators': [200, 500],
    'max_features': ['sqrt'],
    'max_depth' : [4,5,100],
    'criterion' :['gini', 'entropy']
}