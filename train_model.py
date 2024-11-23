import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)
# TODO: load the cencus.csv data
project_path = os.getcwd()
data_path = os.path.join(project_path, "data", "census.csv")
print(data_path)
data = pd.read_csv('data/census.csv')
cleaned_data = data.copy()

# TODO: split the provided data to have a train dataset and a test dataset
# Optional enhancement, use K-fold cross validation instead of a train-test split.

#Here we are going to split the dataset into both a training and test set.
train_full, test = train_test_split(cleaned_data, test_size = 0.20, random_state = 42)

# First, we will define the number of folds we want.
n_splits = 5
kf = KFold(n_splits = n_splits, shuffle = True, random_state = 42)

#Next, we need to make a Categorical features list
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

#Here we will collect the metrecs for each fold.
all_metrics = []

#Now we are going to process the dataset before the k-fold splitting.
X, y, encoder, lb = process_data(
    cleaned_data,
    categorical_features = cat_features,
    label = "salary",
    training = True
)

#Here is the K-fold cross validation.
for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

#Now we will train the model on the fold.
    model = train_model(X_train, y_train)

    preds = inference(model, X_val)

#Here we are going to compute the metrics.
    p, r, fb = compute_model_metrics(y_val, preds)
    all_metrics.append((p, r, fb))

#Now we average the metgrics across all the folds.
avg_precision = sum(m[0] for m in all_metrics) / n_splits
avg_recall = sum(m[1] for m in all_metrics) / n_splits
avg_f1 = sum(m[2] for m in all_metrics) / n_splits

print(f"Average Precision: {avg_precision:.4f} | Average Recall: {avg_recall:.4f} | Average F1: {avg_f1:.4f}")

#Here we are going to process the test dataset.
X_test, y_test, _, _ = process_data(
    test,
    categorical_features = cat_features,
    label = 'salary',
    training = False,
    encoder = encoder,
    lb = lb,
)

preds = inference(model, X_test)

# Now we can calculate and print the metrics on the test set.
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Test Precision: {p:.4f} | Test Recall: {r:.4f} | Test F1: {fb:.4f}")

#Here we will test the performance on the categorical slices.
for col in cat_features:
    for slicevalue in sorted(test[col].unique()):
        count = test[test[col] == slicevalue].shape[0]

        test_slice = test[test[col] == slicevalue]
        X_slice, y_slice, _, _ = process_data(
            test_slice,
            categorical_features = cat_features,
            label = 'salary',
            training = False,
            encoder = encoder,
            lb = lb
        )
        preds_slice = inference(model, X_slice)
        p, r, fb = performance_on_categorical_slice(
            test_slice,
            col,
            slicevalue,
            model,
            y_slice,
            encoder,
            lb
        )
        with open("slice_output.txt", "a") as f:
            print(f"{col}: {slicevalue}, Count: {count:,}", file = f)
            print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}", file = f)


#splitting the data into a test and train set.
train_full, test = train_test_split(cleaned_data, test_size=0.20)

#Splitting the data into a train and validation set.
train, val = train_test_split(train_full, test_size=0.25)
train, test = None, None# Your code here

model = train_model(X_train, y_train)


# DO NOT MODIFY
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

# TODO: use the process_data function provided to process the data.
X_train, y_train, encoder, lb = process_data(
    # your code here
    # use the train dataset 
    # use training=True
    # do not need to pass encoder and lb as input
    )

X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# TODO: use the train_model function to train the model on the training dataset
model = train_model(X_train, y_train)


# save the model and the encoder
model_path = os.path.join(project_path, "model", "model.pkl")
save_model(model, model_path)
encoder_path = os.path.join(project_path, "model", "encoder.pkl")
save_model(encoder, encoder_path)

# load the model
model = load_model(
    model_path
) 

# TODO: use the inference function to run the model inferences on the test dataset.
preds = model.predict(X)

# Calculate and print the metrics
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# TODO: compute the performance on model slices using the performance_on_categorical_slice function
# iterate through the categorical features
for col in cat_features:
    # iterate through the unique values in one categorical feature
    for slicevalue in sorted(test[col].unique()):
        count = test[test[col] == slicevalue].shape[0]
        p, r, fb = performance_on_categorical_slice(
            test_slice = test[test[col]],
            preds_slice = inference(model, X_slice),
            y_slice = test_slice['Salary']
            # your code here
            # use test, col and slicevalue as part of the input
        )
        with open("slice_output.txt", "a") as f:
            print(f"{col}: {slicevalue}, Count: {count:,}", file=f)
            print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}", file=f)
