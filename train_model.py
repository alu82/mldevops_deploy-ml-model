import logging
import pandas as pd

from sklearn.model_selection import train_test_split

from ml.data import process_data, slice_data
from ml.model import (
    train_model, inference, save_training_artifacts, compute_model_metrics
)

logging.basicConfig(
    filename="./log/train_model.log",
    format="%(levelname)s:%(asctime)s:%(message)s",
    level=logging.INFO,
    filemode='w'
)


def log_slice_metrics(data, model, encoder, lb, categorical_features):
    '''
    This methods calculates the performance of a model for slices defined
    by the values of categorical features.
    '''
    f = open("./log/slice_output.txt", "w")
    for feature, cls, slice in slice_data(
            data=data, categorical_features=cat_features):
        X, y, _, _ = process_data(
            slice, categorical_features=categorical_features,
            label="salary", training=False,
            encoder=encoder, lb=lb
        )
        y_pred = inference(model, X)
        precision, recall, fbeta = compute_model_metrics(y, y_pred)
        log_string = (
            f"Metrics for slice {feature}|{cls}: "
            f"Precision {precision}, Recall {recall}, fbeta {fbeta}"
        )
        logging.info(log_string)
        f.write(f"{log_string}\n")
    f.close()


# Add code to load in the data.
logging.info("Reading data from file.")
data = pd.read_csv('./data/census.csv')
logging.info(
    "Reading data from file successful. Shape of the dataset: %s.",
    data.shape)

# Optional enhancement, use K-fold cross validation instead of a
# train-test split.
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

# Proces the test data with the process_data function. Use the fitted
# models from training to do the data preprocessing
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features,
    label="salary", training=False,
    encoder=encoder, lb=lb
)

# Train and save a model.
model = train_model(X_train, y_train)
y_test_pred = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, y_test_pred)
logging.info(
    "Overall metrics: Precision %s, Recall %s, fbeta %s",
    precision,
    recall,
    fbeta)
log_slice_metrics(data, model, encoder, lb, cat_features)
save_training_artifacts("./model", model, encoder, lb)
