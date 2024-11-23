import pickle
from sklearn.metrics import fbeta_score, precision_score, recall_score
from ml.data import process_data
from sklearn.linear_model import LogisticRegression

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta = 1, zero_division = 1)
    precision = precision_score(y, preds, zero_division = 1)
    recall = recall_score(y, preds, zero_division = 1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds

def save_model(model, path):
    """ Serializes model to a file.

    Inputs
    ------
    model
        Trained machine learning model or OneHotEncoder.
    path : str
        Path to save pickle file.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(instance, f)

def load_model(path):
    """ Loads pickle file from `path` and returns it."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)




def performance_on_categorical_slice(
    data,
    column_name,             
    slice_value,       
    model,             
    label,             
    categorical_features,  
    encoder,           
    lb                 
):
    X_slice = data[data[column_name] == slice_value]
    
    if label is not None:
        y_slice = X_slice[label]
        X_slice = X_slice.drop(columns=[label])
    else:
        y_slice = pd.Series(dtype=int)
    
    X_processed = process_data(
        X_slice,
        categorical_features=categorical_features,
        training=False,
        encoder=encoder,
        lb=lb
    )
    
    preds = model.predict(X_processed)
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)
    return precision, recall, fbeta