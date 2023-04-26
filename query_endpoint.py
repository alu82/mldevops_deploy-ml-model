'''
Module to test a deployed live endpoint.
Author: alu82
Date: April 2023
'''

import requests
import json

ENDPOINT_URL = "https://mldevops-deploy-ml-model.onrender.com"

request = {
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

response = requests.post(f'{ENDPOINT_URL}/adult/', data=json.dumps(request))

print(response.status_code)
print(response.json())
