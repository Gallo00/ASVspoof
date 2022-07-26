from tkinter import Variable
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import numpy as np
from typing import Union
from constants import ROWS_CREATION_MODELS, FEATURES, VARIABLES_TO_MANTAIN
from EER import curve_frr_far, eer

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def get_training_test_sets() -> Union[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data1 = pd.read_csv('./datasetPart00l.csv')
    data2 = pd.read_csv('./datasetPart01l.csv')
    data3 = pd.read_csv('./datasetPart02l.csv')
    data4 = pd.read_csv('./datasetPart03l.csv')

    data = pd.concat([data1,data2,data3,data4], axis=0)

    data = data.drop('file',axis=1)
    #data = data.drop(VARIABLES_TO_DROP,axis=1)
    data = data[['label'] + VARIABLES_TO_MANTAIN]
    data = data.fillna(0)

    data1 = data[data['label'] == 'spoof']
    data2 = data[data['label'] == 'bonafide']

    #create a balanced situation
    data = pd.concat([data1.head(10000),data2.head(10000)],axis=0)
    
    #data = data.sample(ROWS_CREATION_MODELS)

    X = data.copy()
    X = X.drop('label',axis=1)
    Y = data['label']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=True, test_size=0.3)
    return X_train, X_test, Y_train, Y_test



def model_creation(classifier: any, labels: list) -> Union[any, np.float64, ConfusionMatrixDisplay ]:
    [BONAFIDE, SPOOF] = labels
    X_train, X_test, Y_train, Y_test = get_training_test_sets()

    #scale data
    """
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    """

    # set neighbors to 2 if the classifier is KNN
    model_params = {}
    if classifier == KNeighborsClassifier:
        model_params["n_neighbors"] = 2
    if classifier == SVC:
        model_params["probability"] = True
    model = classifier(**model_params)
    if classifier == MultinomialNB:
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    predictions_proba = model.predict_proba(X_test)

    bonafide_probabilities = []
    for couple in predictions_proba:
        bonafide_probabilities.append(couple[0])
    targets = []
    for pred in predictions:
        if pred == 'bonafide':
            targets.append(0)
        else:
            targets.append(1)
    targets = np.asarray(targets)
    bonafide_probabilities = np.asarray(bonafide_probabilities)
    th, frr, far = curve_frr_far(targets=targets, genuine_probabilities=bonafide_probabilities, genuine_label=0)
    EER, fEER = eer(th,frr,far)
    print("EER:" , EER)

    acc = accuracy_score(Y_test, predictions)
    rec = recall_score(Y_test, predictions, pos_label=BONAFIDE)
    prec = precision_score(Y_test, predictions, pos_label=BONAFIDE)
    conf_matrix = confusion_matrix(Y_test,predictions)
    cm_display = ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels = [BONAFIDE, SPOOF])
    print('acc:', acc)
    print('rec:', rec)
    print('prec:', prec)

    return model, acc, prec, rec, EER, cm_display, fEER


