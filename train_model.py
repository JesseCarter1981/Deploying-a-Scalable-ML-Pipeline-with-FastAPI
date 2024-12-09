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
# load the cencus.csv data
project_path = os.getcwd()
data_path = os.path.join(project_path, 'data', 'census.csv')
print(data_path)
data = pd.read_csv('data/census.csv')
cleaned_data = data.copy()

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
    label = 'salary',
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

print(f'Average Precision: {avg_precision:.4f} | Average Recall: {avg_recall:.4f} | Average F1: {avg_f1:.4f}')

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
for col in cat_features :
    for slicevalue in sorted(test[col].unique()):
        count = test[test[col] == slicevalue].shape[0]

        test_slice = test[test[col] == slicevalue]
        X_slice, y_slice, _, _ = process_data(
            test_slice,
            categorical_features = cat_features ,
            label = 'salary',
            training = False,
            encoder = encoder,
            lb = lb
        )

        preds_slice = inference(model, X_slice)
        p, r, fb = performance_on_categorical_slice(
            data,
            col,
            slicevalue,
            cat_features,
            'salary',
            encoder,
            lb,
            model
        )

        with open("slice_output.txt", "a") as f:
            print(f"{col}: {slicevalue}, Count: {count:,}", file = f)
            print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}", file = f)


# Train final model with the entire training data
X_train, y_train, encoder, lb = process_data(
    train_full,
    categorical_features = cat_features,
    label = 'salary',
    training = True
)

model = train_model(X_train, y_train)

# Save the model and encoder
model_path = os.path.join(project_path, 'model', 'model.pkl')
save_model(model, model_path)
encoder_path = os.path.join(project_path, 'model', 'encoder.pkl')
save_model(encoder, encoder_path)

# Load the model for inference
model = load_model(model_path)

# Process the test set for final evaluation
X_test, y_test, _, _ = process_data(
    test,
    categorical_features = cat_features,
    label='salary',
    training=False,
    encoder=encoder,
    lb=lb,
)

# Use the model to make predictions
preds = model.predict(X_test)

# Calculate and print the metrics
p, r, fb = compute_model_metrics(y_test, preds)
print(f'recision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}')

# Compute performance on categorical slices
for col in cat_features:
    for slicevalue in sorted(test[col].unique()):
        count = test[test[col] == slicevalue].shape[0]

        # Get the slice for this categorical feature value
        test_slice = test[test[col] == slicevalue]

        # Process slice data
        X_slice, y_slice, _, _ = process_data(
            test_slice,
            categorical_features = cat_features,
            label='salary',
            training=False,
            encoder=encoder,
            lb=lb
        )

        # Make predictions for the slice
        preds_slice = inference(model, X_slice)

        # Calculate performance metrics for the slice
        p, r, fb = performance_on_categorical_slice(
            test_slice, col, slicevalue, cat_features, 'salary', encoder, lb, model
        )

        # Save the results to a file
        with open('slice_output.txt', 'a') as f:
            print(f'{col}: {slicevalue}, Count: {count:,}', file=f)
            print(f'Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}', file=f)