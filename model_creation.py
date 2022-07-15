from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from constants import ROWS_CREATION_MODELS

def get_training_test_sets():
    data = pd.read_csv('./dataset.csv')
    data = data.dropna()
    data = data.drop('file',axis=1)

    """
    dataframe_bonafide = data.loc[data['label'] == 'bonafide']
    dataframe_spoof = data.loc[data['label'] == 'spoof']

    dataframe_bonafide = dataframe_bonafide.sample(ROWS_CREATION_MODELS)
    dataframe_spoof = dataframe_spoof.sample(ROWS_CREATION_MODELS)
    #data = data.sample(ROWS_CREATION_MODELS)

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

    model = None 

    if classifier == KNeighborsClassifier:
        model = classifier(n_neighbors=2)
    elif classifier == MultinomialNB:
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        model = classifier()
    else: 
        model = classifier()

    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    acc = accuracy_score(Y_test, predictions)
    print(acc)

    return model, acc 
