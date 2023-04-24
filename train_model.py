import logging
import pandas as pd

from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import train_model, inference, save_training_artifacts, compute_model_metrics

logging.basicConfig(
    filename="./log/train_model.log",
    format="%(levelname)s:%(asctime)s:%(message)s",
    level=logging.INFO,
    filemode='w'
)

# Add code to load in the data.
logging.info("Reading data from file.")
data = pd.read_csv('./data/census.csv')
logging.info("Reading data from file successful.")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function. Use the fitted models from training to do the data preprocessing
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# Train and save a model.
model = train_model(X_train, y_train)
y_test_pred = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, y_test_pred)
logging.info("Precision %s, Recall %s, fbeta %s", precision, recall, fbeta)
save_training_artifacts("./model", model, encoder, lb)


