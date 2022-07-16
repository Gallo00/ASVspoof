from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
from constants import ROWS_CREATION_MODELS, VARIABLES_TO_DROP

def get_training_test_sets():
    data = pd.read_csv('./dataset.csv')
    data = data.dropna()
    data = data.drop('file',axis=1)
    data = data.drop(VARIABLES_TO_DROP,axis=1)

    """
    dataframe_bonafide = data.loc[data['label'] == 'bonafide']
    dataframe_spoof = data.loc[data['label'] == 'spoof']
    dataframe_bonafide = dataframe_bonafide.sample(ROWS_CREATION_MODELS)
    dataframe_spoof = dataframe_spoof.sample(ROWS_CREATION_MODELS)
    data = dataframe_bonafide.append(dataframe_spoof, ignore_index=True)
    """
    data = data.sample(ROWS_CREATION_MODELS)
    
    X = data.copy()
    X = X.drop('label',axis=1)
    Y = data.pop('label')

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 100)
    return X_train, X_test, Y_train, Y_test



def model_creation(classifier):
    X_train, X_test, Y_train, Y_test = get_training_test_sets()
    scaler = MinMaxScaler() # necessary step to use MultinomialNB
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = None 

    if classifier == KNeighborsClassifier:
        model = classifier(n_neighbors=2)
    else: 
        model = classifier()

    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    acc = accuracy_score(Y_test, predictions)
    rec = recall_score(Y_test, predictions, pos_label="bonafide")
    prec = precision_score(Y_test, predictions, pos_label="bonafide")
    conf_matrix = confusion_matrix(Y_test,predictions)
    cm_display = ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels = ['bonafide','spoof'])
    print('acc:', acc)
    print('rec:', rec)
    print('prec:', prec)

    return model, acc, cm_display
