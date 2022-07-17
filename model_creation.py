from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import numpy as np
from typing import Union
from collinearity import SelectNonCollinear
from constants import ROWS_CREATION_MODELS, VARIABLES_TO_DROP, FEATURES

selector = SelectNonCollinear(0.4)


def get_training_test_sets() -> Union[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv('./dataset.csv')
    data = data.dropna()
    data = data.drop('file',axis=1)
    data = data.drop(VARIABLES_TO_DROP,axis=1)

    
    dataframe_bonafide = data.loc[data['label'] == 'bonafide']
    dataframe_spoof = data.loc[data['label'] == 'spoof']
    dataframe_bonafide = dataframe_bonafide.sample(int(ROWS_CREATION_MODELS / 8))
    dataframe_spoof = dataframe_spoof.sample(ROWS_CREATION_MODELS)

    data = pd.concat([dataframe_bonafide, dataframe_spoof], axis=0)
    print(data.head())
    data.sample(frac=1).reset_index(drop=True)
    
    #data = data.sample(ROWS_CREATION_MODELS)

    print(data.loc[data['label'] == 'bonafide'].head())
    print(data.loc[data['label'] == 'spoof'].head())

    X = data.copy()
    X = X.drop('label',axis=1)
    Y = data.pop('label')

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=True)
    return X_train, X_test, Y_train, Y_test



def model_creation(classifier: any, labels: list) -> Union[any, np.float64, ConfusionMatrixDisplay ]:
    [BONAFIDE, SPOOF] = labels
    X_train, X_test, Y_train, Y_test = get_training_test_sets()

    #scale data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # remove collinear variables
    selector.fit(X_train,Y_train)
    mask = selector.get_support()
    features = FEATURES.copy() 
    for x in VARIABLES_TO_DROP:
        features.remove(x)
    X_train = pd.DataFrame(X_train[:,mask],columns = np.array(features)[mask])
    X_test = pd.DataFrame(X_test[:,mask],columns = np.array(features)[mask])

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
