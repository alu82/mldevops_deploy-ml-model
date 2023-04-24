# Model Card
For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The model was created by alu82. It is a RandomForrestClassifier with the following parameters:
- n_estimators = 100 (default value)
- random_state = 42 (for reproducibility)
scikit-learn with version 1.2.2 was used.

## Intended Use
The model should predict the expected salary based on some attributes that represent personal information. It classifies the income into 2 classes: <=50K and >50K.

It should be used for non critical use cases. A use case that probably won't change a persons life is the type of ads shown to him/her.

## Training Data
The data was obtained from a public source (UCI Machine Learning Repository). More information on the raw dataset can be found here: https://archive.ics.uci.edu/ml/datasets/census+income.

The dataset has 32561 rows and 15 columns, where our label (salary) is one of them. The dataset has been cleaned manually by replacing every occurence of ```, ``` with ```,```. To use the data for training a One Hot Encoder was used on the features and a label binarizer was used on the labels.

To train the model 80% of the data has been used.

## Evaluation Data
For the evaluation of the model 20% of the original dataset has been used. 

## Metrics
As metrics precision, recall and fbeta has been used. In the current version the model has the following performance:
|Precision|Recall|fbeta|
|---------|------|-----|
|0.735|0.609|0.666|

## Ethical Considerations
The model has been trained on very sensitive personal data and a relative small dataset. Therefore there might be some bias within the data. The model should not be used to make critical decisions on a persons life.

## Caveats and Recommendations
The model has been trained on default parameters. It is recommended to do a hyperparameter tuning to increase the models performance. Also it should be investigated if the model is biased and fair, and if not, corresponding measurements has to be taken.