from sklearn import discriminant_analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#https://newbedev.com/fisher-s-linear-discriminant-in-python

# formula here https://compbio.soe.ucsc.edu/genex/genexTR2html/node12.html
def fisher_criterion(population1, population2):
    return abs(np.mean(population1) - np.mean(population2)) / (np.var(population1) + np.var(population2))

data1 = pd.read_csv('./datasetPart00l.csv')
data2 = pd.read_csv('./datasetPart01l.csv')
data3 = pd.read_csv('./datasetPart02l.csv')
data4 = pd.read_csv('./datasetPart03l.csv')

data = pd.concat([data1,data2,data3,data4], axis=0)

data = data.drop('file',axis=1)

var = 'bit_rate'

data_var = data[['label', var]]
data_var = data_var.fillna(0)

data_spoof = data_var[data['label'] == 'spoof']
data_bonafide = data_var[data['label'] == 'bonafide']

ROWS = 1000
data_var = pd.concat([data_spoof.sample(ROWS),data_bonafide.sample(ROWS)],axis=0)

data_var.reset_index(inplace = True, drop = True)

X = data_var.drop('label', axis=1)
y = data_var.drop(var, axis=1)



X.columns = ['feat_0']
y.columns = ['labels']

tot = pd.concat([X,y], axis=1)
# calculate class means
class_means = tot.groupby('labels').mean()
total_mean = X.mean()

x_mi = tot.transform(lambda x: x - class_means.loc[x['labels']], axis=1).drop('labels', 1)

def kronecker_and_sum(df, weights):
    S = np.zeros((df.shape[1], df.shape[1]))
    for idx, row in df.iterrows():
        row = pd.DataFrame(row)
        x_m = row.values.reshape(df.shape[1],1)
        try:
            dot_prod = np.dot(x_m, x_m.T)
            S = S + weights[idx]*dot_prod
        except:
            dfaa = 1
    return S
# Each x_mi is weighted with 1. Now we use the kronecker_and_sum function to calculate the within-class scatter matrix S_w
tot_rows = ROWS*2    
S_w = kronecker_and_sum(x_mi, tot_rows*[1])

mi_m = class_means.transform(lambda x: x - total_mean, axis=1)
# Each mi_m is weighted with the number of observations per class which is 50 for each class in this example. We use kronecker_and_sum to calculate the between-class scatter matrix.

S_b=kronecker_and_sum(mi_m, 2*[ROWS])

S_w = S_w.astype(float)
inv = np.linalg.inv(S_w)

eig_vals, eig_vecs = np.linalg.eig(inv.dot(S_b))

W = eig_vecs[:, :2]
X_trafo = np.dot(X, W)
tot_trafo = pd.concat([pd.DataFrame(X_trafo, index=range(len(X_trafo))), y], 1)
# plot the result
tot_trafo.columns = [var, 'labels']

red = ['red'] * ROWS
blue = ['blue'] * ROWS

colors = red + blue
tot_trafo.plot.scatter(x=var, y=1,c=colors, colormap='viridis')
plt.show()


"""
#same results
lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=1)
X_trafo_sk = lda.fit_transform(X,y)
red = ['red'] * ROWS
blue = ['blue'] * ROWS
colors = red + blue

pd.DataFrame(np.column_stack((X_trafo_sk, y))).plot.scatter(x=var, y=1,c=colors, colormap='viridis')

plt.show()
"""