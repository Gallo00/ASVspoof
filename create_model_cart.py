from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd


data = pd.read_csv('./dataset.csv')
data = data.dropna()
data = data.drop('file',axis=1)

rows = 1000
data = data.sample(rows)

X = data.copy()
X = X.drop('label',axis=1)
Y = data.pop('label')

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 100)

clf = DecisionTreeClassifier(
    criterion='gini', 
    splitter='best', 
    max_depth=None, 
    min_samples_split=2, 
    min_samples_leaf=1, 
    min_weight_fraction_leaf=0.0, 
    max_features=None, 
    random_state=None, 
    max_leaf_nodes=None, 
    min_impurity_decrease=0.0, 
    class_weight=None, 
    ccp_alpha=0.0
)

clf.fit(X_train, Y_train)

predictions = clf.predict(X_test)

print(accuracy_score(Y_test, predictions))

