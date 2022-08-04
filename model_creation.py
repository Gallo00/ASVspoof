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
from constants import  VARIABLES_TO_MANTAIN
from metrics_eer_accperclass import compute_eer, compute_accperclass


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

    ROWS = 10000
    #create a balanced situation
    data = pd.concat([data1.sample(ROWS),data2.sample(ROWS)],axis=0) #balanced
    
    
    #data = pd.concat([data1, data2], axis=0) #unbalanced
    #data = data.sample(2*ROWS) #unbalanced
    

    X = data.copy()
    X = X.drop('label',axis=1)
    Y = data['label']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=True, test_size=0.3)
    return X_train, X_test, Y_train, Y_test



def model_creation(classifier: any, labels: list) -> Union[any, np.float64, ConfusionMatrixDisplay ]:
    [BONAFIDE, SPOOF] = labels
    X_train, X_test, Y_train, Y_test = get_training_test_sets()

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

    acc = accuracy_score(Y_test, predictions)
    rec = recall_score(Y_test, predictions, pos_label=BONAFIDE)
    prec = precision_score(Y_test, predictions, pos_label=BONAFIDE)
    conf_matrix = confusion_matrix(Y_test,predictions)
    #get TP etc. to calculate EER
    TP = conf_matrix[0][0]
    FN = conf_matrix[0][1]
    FP = conf_matrix[1][0]
    TN = conf_matrix[1][1]
    EER = compute_eer(TP, FN, FP, TN)
    print("EER:",EER)

    acc_per_class = compute_accperclass(TP, FN, FP, TN)

    cm_display = ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels = [BONAFIDE, SPOOF])
    print('acc:', acc)
    print('acc_per_class', acc_per_class)
    print('rec:', rec)
    print('prec:', prec)

    return model, acc, acc_per_class, prec, rec, EER, cm_display


