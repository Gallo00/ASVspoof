import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay
from metrics_eer_accperclass import compute_accperclass, compute_eer
import matplotlib.pyplot as plt

BONAFIDE = 'bonafide'
SPOOF = 'spoof'

ACC_INDEX = 0
ACC_PER_CLASS_INDEX = 1
PREC_INDEX = 2
REC_INDEX = 3
EER_INDEX = 4
CM_DISPLAY_INDEX = 5
M_S_INDEX = 6
M_B_INDEX = 7


def save_files(model: list, type: str) -> None:
    model[CM_DISPLAY_INDEX].plot()

    #model_path = './models_test_all_ds/Naive_mean/' + type
    #model_path = './models_unbalanced/Naive_mean/' + type #unbalanced
    model_path = './models/Naive_mean/' + type #balanced
    plt.savefig(model_path + '/conf_matrix.png')
    metrics = {
        "EER": model[EER_INDEX],
        "accuracy": model[ACC_INDEX],
        "accuracy_per_class": model[ACC_PER_CLASS_INDEX],
        "precision": model[PREC_INDEX],
        "recall": model[REC_INDEX]
    }
    with open(model_path + '/metrics.yml', 'w') as f: 
        for key, value in metrics.items(): 
            f.write('%s: %s\n' % (key, value))

    means = {
        "mean_spoof": model[M_S_INDEX],
        "mean_bonafide": model[M_B_INDEX],
    }
    with open(model_path + '/means.yml', 'w') as f: 
        for key, value in means.items(): 
            f.write('%s: %s\n' % (key, value))

    if type == 'mean':
        line_md = "| **Naive_mean** "
        for k, v in metrics.items():
            line_md += "|" + str(round(v,4))
        print(line_md + " |")

data1 = pd.read_csv('./datasetPart00l.csv')
data2 = pd.read_csv('./datasetPart01l.csv')
data3 = pd.read_csv('./datasetPart02l.csv')
data4 = pd.read_csv('./datasetPart03l.csv')

data = pd.concat([data1,data2,data3,data4], axis=0)

data = data.drop('file',axis=1)

var = 'bit_rate'

print("naive classifier based on: ", var)
data_var = data[['label', var]]
data_var = data_var.fillna(0)

data_spoof = data_var[data['label'] == 'spoof']
data_bonafide = data_var[data['label'] == 'bonafide']

naive_models = []

for i in range(10):
    ROWS = 10000
    data_var = pd.concat([data_spoof.sample(ROWS),data_bonafide.sample(ROWS)],axis=0) #balanced
    #data_var = pd.concat([data_spoof,data_bonafide],axis=0) #test on all dataset

    #data_var = pd.concat([data_spoof, data_bonafide], axis=0) #unbalanced
    #data_var = data_var.sample(2*ROWS) #unbalanced

    var_spoof = data_spoof[var].to_numpy() 
    var_bonafide = data_bonafide[var].to_numpy()


    data_var.insert(2,'pred_label','')

    data_var.reset_index(inplace = True, drop = True)

    mean_bonafide = var_bonafide.mean()
    mean_spoof = var_spoof.mean()


    for i in range(len(data_var)):
        dist_spoof = abs(data_var.loc[i, var] - mean_spoof)
        dist_bonafide = abs(data_var.loc[i, var] - mean_bonafide)

        if dist_spoof < dist_bonafide:
            data_var.loc[i, 'pred_label'] = 'spoof'
        else:
            data_var.loc[i, 'pred_label'] = 'bonafide'
    
    y_true = data_var['label'].to_numpy() 
    y_pred = data_var['pred_label'].to_numpy()


    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, pos_label=BONAFIDE)
    prec = precision_score(y_true, y_pred, pos_label=BONAFIDE)
    conf_matrix = confusion_matrix(y_true,y_pred)
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

    mod = [acc, acc_per_class, prec, rec, EER, cm_display, mean_spoof, mean_bonafide]
    naive_models.append(mod)


# calculate the mean of EER
mean_eer = 0.0
for mod in naive_models:
    mean_eer += mod[EER_INDEX]
mean_eer = mean_eer / len(naive_models)

# search the best, worst and mean model
best = naive_models[0]
mean = naive_models[0]
worst = naive_models[0]

dist = abs(mean[EER_INDEX] - mean_eer)

for mod in naive_models:
    if mod[EER_INDEX] < best[EER_INDEX]:
        best = mod
    elif mod[EER_INDEX] > worst[EER_INDEX]:
        worst = mod

    dist_actual = abs(mod[EER_INDEX] - mean_eer)
    if dist_actual < dist:
        dist = dist_actual
        mean = mod

# SAVE BEST MODEL
save_files(best, 'best')

# SAVE MEAN MODEL 
save_files(mean, 'mean')

# SAVE WORST MODEL 
save_files(worst, 'worst')




