from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import numpy as np
from typing import Union
from constants import ROWS_CREATION_MODELS, VARIABLES_TO_DROP, FEATURES


def get_training_test_sets() -> Union[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv('./dataset.csv')
    data = data.drop('file',axis=1)
    data = data.drop(VARIABLES_TO_DROP,axis=1)
    data = data.fillna(0)

    data = data.sample(ROWS_CREATION_MODELS)

    X = data.copy()
    X = X.drop('label',axis=1)
    Y = data['label']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=True)
    return X_train, X_test, Y_train, Y_test



def model_creation(classifier: any, labels: list) -> Union[any, np.float64, ConfusionMatrixDisplay ]:
    [BONAFIDE, SPOOF] = labels
    X_train, X_test, Y_train, Y_test = get_training_test_sets()

    #scale data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    # set neighbors to 2 if the classifier is KNN
    model_params = {}
    if classifier == KNeighborsClassifier:
        model_params["n_neighbors"] = 2
    model = classifier(**model_params)
    
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    acc = accuracy_score(Y_test, predictions)
    rec = recall_score(Y_test, predictions, pos_label=BONAFIDE)
    prec = precision_score(Y_test, predictions, pos_label=BONAFIDE)
    conf_matrix = confusion_matrix(Y_test,predictions)
    cm_display = ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels = [BONAFIDE, SPOOF])
    print('acc:', acc)
    print('rec:', rec)
    print('prec:', prec)

    return model, acc, cm_display
