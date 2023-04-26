from fastapi.testclient import TestClient
import pytest

# import api app
from main import app, Adult

# Instantiate the testing client with our app.
client = TestClient(app)

@pytest.fixture(name="adult_high_salary")
def create_adult_high_salary():
    '''
    Fixture to create a json for an adult with an high predicted income.
    '''
    return {
        "age": 43,
        "workclass": "Federal-gov",
        "fnlgt": 410867,
        "education": "Doctorate",
        "education_num": 16,
        "marital_status": "Never-married",
        "occupation": "Exec-managerial",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 45,
        "native_country": "United-States"
    }

@pytest.fixture(name="adult_low_salary")
def create_adult_low_salary():
    '''
    Fixture to create a json forn an adult with a low predicted income.
    '''
    return {
        "age": 43,
        "workclass": "Federal-gov",
        "fnlgt": 410867,
        "education": "Bachelor",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Exec-managerial",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 45,
        "native_country": "United-States"
    }

def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Hello World!"}

def test_prediction_high_salary(adult_high_salary):
    r = client.post("/adult/", json=adult_high_salary)
    assert r.status_code == 200
    assert r.json()["salary-prediction"] == 1

def test_prediction_low_salary(adult_low_salary):
    r = client.post("/adult/", json=adult_low_salary)
    assert r.status_code == 200
    assert r.json()["salary-prediction"] == 0
