import os
import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score
from ml.model import train_model, inference, save_model, load_model, performance_on_categorical_slice, process_data

project_path = os.getcwd()
data_path = os.path.join(project_path, 'data', 'census.csv')

cat_features = [
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country',
]

from sklearn.metrics import precision_score, recall_score, accuracy_score
def compute_model_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    return precision, recall, accuracy

@pytest.fixture
def model_data():
    train = pd.read_csv(data_path)
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features = cat_features, label = 'salary', training = True)

    model = train_model(X_train, y_train)
    return model, X_train, y_train

def test_compute_model_metrics(model_data):
    model, X_train, y_train = model_data
    
    
    preds = inference(model, X_train)
    metrics = compute_model_metrics(y_train, preds)

    assert len(metrics) == 3
    assert isinstance(metrics, tuple)

    for metric in metrics:
        assert 0 <= metric <= 1

def test_inference(model_data):
    model, X_train, _ = model_data
    
    preds = inference(model, X_train)

    assert len(preds) == len(X_train)
    
    assert np.all((preds == 0) | (preds == 1))

def test_train_model(model_data):
    model, _, _ = model_data
    
    assert isinstance(model, RandomForestClassifier)