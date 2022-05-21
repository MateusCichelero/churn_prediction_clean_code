"""
Pytest testing configuration file.

Author: Mateus Cichelero
Date: May 2022
"""
import pytest

def df_plugin():
  return None
# creating a Dataframe object 'pytest.df' in Namespace
def pytest_configure():
  pytest.df = df_plugin()
  pytest.x_train = df_plugin()
  pytest.x_test = df_plugin()
  pytest.y_train = df_plugin() 
  pytest.y_test = df_plugin()