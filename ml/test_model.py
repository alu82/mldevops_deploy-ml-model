'''
Module that contains all unit test for the modelling module.
'''
import pytest
import numpy as np
import sklearn
from .model import train_model, compute_model_metrics, inference


@pytest.fixture(name="X_train")
def x_train_fix():
    '''
    Fixture to get some random training data
    (has nothing to do with census data)
    '''
    return np.random.rand(100, 20)


@pytest.fixture(name="y_train")
def y_train_fix():
    '''
    Fixture to get some random training data
    (has nothing to do with census data)
    '''
    return np.random.randint(0, 2, 100)


@pytest.fixture(name="y_pred")
def y_pred_fix():
    '''
    Fixture to get some random training data
    (has nothing to do with census data)
    '''
    return np.random.randint(0, 2, 100)


@pytest.fixture(name="model")
def model_fix(X_train, y_train):
    '''
    Fixture to get some default model
    '''
    return train_model(X_train, y_train)


def test_train_model(X_train, y_train):
    '''
    Test correct type of trained model
    '''
    model = train_model(X_train, y_train)
    assert isinstance(model, sklearn.ensemble.RandomForestClassifier)


def test_compute_model_metrics(y_train, y_pred):
    '''
    Tests correct return types and range of compute_model_metrices method
    '''
    precision, recall, fbeta = compute_model_metrics(y_train, y_pred)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
    assert 0 <= precision and precision <= 1
    assert 0 <= recall and recall <= 1
    assert 0 <= fbeta and fbeta <= 1


def test_inference(model, X_train):
    '''
    Tests correct return type and shape
    '''
    pred = inference(model, X_train)
    assert isinstance(pred, np.ndarray)
    assert len(X_train) == len(pred)
