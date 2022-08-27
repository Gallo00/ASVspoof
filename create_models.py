from model_creation import model_creation
import pickle
import matplotlib.pyplot as plt
from constants import CLASSIFIERS

MOD_INDEX = 0
ACC_INDEX = 1
ACC_PER_CLASS_INDEX = 2
PREC_INDEX = 3
REC_INDEX = 4
EER_INDEX = 5
CM_DISPLAY_INDEX = 6

lines_md = []
lines_md.append("|Model|EER|Accuracy|Accuracy per class|Precision|Recall|")
lines_md.append("|-------------------------------|-----------|------------|--------------------|-----------|------------|")


def save_files(model: list, type: str) -> None:
    model[CM_DISPLAY_INDEX].plot()

    model_path = './models_test_all_ds/' + classifier.__name__ + '/' + type
    #model_path = './models_unbalanced/' + classifier.__name__ + '/' + type #unbalanced
    #model_path = './models/' + classifier.__name__ + '/' + type #balanced

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

    if type == 'mean':
        line_md = "| **" + classifier.__name__ + "** "
        for k, v in metrics.items():
            line_md += "|" + str(round(v,4))
        lines_md.append(line_md + "|")
    pickle.dump(model[MOD_INDEX], open(model_path + '/model.pkl', 'wb'))


for classifier in CLASSIFIERS:
    print(classifier.__name__)

    # try to create some models per type
    models_metrics = []
    for i in range(10):
        model, acc, acc_per_class, prec, rec, eer, cm_display = model_creation(classifier, ['bonafide', 'spoof'])
        models_metrics.append([model, acc, acc_per_class, prec, rec, eer, cm_display])
    
    # calculate the mean of EER
    mean_eer = 0.0
    for mod in models_metrics:
        mean_eer += mod[EER_INDEX]
    mean_eer = mean_eer / len(models_metrics)

    # search the best, worst and mean model
    best_model = models_metrics[0]
    mean_model = models_metrics[0]
    worst_model = models_metrics[0]

    dist = abs(mean_model[EER_INDEX] - mean_eer)

    for mod in models_metrics:
        if mod[EER_INDEX] < best_model[EER_INDEX]:
            best_model = mod
        elif mod[EER_INDEX] > worst_model[EER_INDEX]:
            worst_model = mod

        dist_actual_model = abs(mod[EER_INDEX] - mean_eer)
        if dist_actual_model < dist:
            dist = dist_actual_model
            mean_model = mod

    # this script will save confusion matrix, metrics in a yaml and the model
    # The script will save informations about the best, worst and mean model
    # So we will have 9 files for model


    # SAVE BEST MODEL
    save_files(best_model, 'best')

    # SAVE MEAN MODEL 
    save_files(mean_model, 'mean')

    # SAVE WORST MODEL 
    save_files(worst_model, 'worst')

for l in lines_md:
    print(l)




