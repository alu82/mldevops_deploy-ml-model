'''
Module for creating an api to predict the salary of an Adult.
Author: alu82
Date: April 2023
'''

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from ml.model import load_training_artifacts, inference
from ml.data import process_data

# Instantiate the app.
app = FastAPI()

# load the artifacts used for inferencing
model, encoder, lb = load_training_artifacts("./model")

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


class Adult(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    def to_dataframe(self):
        data = {
            "age": self.age,
            "workclass": self.workclass,
            "fnlgt": self.fnlgt,
            "education": self.education,
            "education-num": self.education_num,
            "marital-status": self.marital_status,
            "occupation": self.occupation,
            "relationship": self.relationship,
            "race": self.race,
            "sex": self.sex,
            "capital-gain": self.capital_gain,
            "capital-loss": self.capital_loss,
            "hours-per-week": self.hours_per_week,
            "native-country": self.native_country,
        }
        return pd.DataFrame([data])

    # creates a config with an example for the generated documentation

    class Config:
        schema_extra = {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education_num": 13,
                "marital_status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital_gain": 2174,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "United-States",
            }
        }

# Define a GET on the specified endpoint.


@app.get("/")
async def say_hello():
    return {"greeting": "Hello World!"}


@app.post("/adult/")
async def create_item(adult: Adult):
    df = adult.to_dataframe()
    X, _, _, _ = process_data(
        df, categorical_features=cat_features,
        training=False, encoder=encoder, lb=lb
    )
    y = inference(model, X)
    return {"salary-prediction": 0 if y[0] == 0 else 1}
