# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity
- Author: Mateus Cichelero
- Date: May 2022

## Project Description

Identification of credit card customers most likely to churn.

Includes a Python package for a machine learning project that follows coding (PEP8) and engineering best practices for implementing software (modular, documented and tested).

The package also has the flexibility to run interactively or from the command-line interface (CLI).

It also introduces a problem data scientists across companies face all the time: How do we identify (and later intervene with) customers who are likely to churn?

## Files and data description

The project's basic folder/file structure is:

```
.
├── churn_notebook.ipynb # Original exploratory notebook that was refactored
├── churn_library.py     # Main module with implemented functions for the pipeline
├── churn_script_logging_and_tests.py # Tests and logs for churn_library
├── requirements.txt     # Python requirements for the project
├── constants.py         # Module that stores constants used in project
├── conftest.py          # Pytest testing configuration file
├── pytest.ini           # Pytest global configurations file
├── sequencediagram.jpeg # Project's sequence diagram
├── README.md            # Provides project overview, and instructions to use the code
├── data                 # Stores the dataset analysed
│   └── bank_data.csv
├── images               # Stores EDA plots and classifiers results 
│   ├── eda
│   └── results
├── logs                 # Stores logs
└── models               # Stores models
```


## Dependencies
- Python (>= 3.8)

## Running Files

First, create a new virtual environment for the project and activate it:

```python

python3 -m venv venv

```

```python

source venv/bin/activate

```

Then, install the libraries listed in requirements.txt:

```python

pip install -r requirements.txt

```

To run the churn prediction data science pipeline:

```python

ipython churn_library.py

```

## Testing and Logging
It is possible to test the project running (logs will be stored in logs/churn_library.log):

```python

ipython churn_script_logging_and_tests.py

```

or simply by running:

```python

pytest

```







