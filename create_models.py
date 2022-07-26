from model_creation import model_creation
import pickle
import matplotlib.pyplot as plt
from constants import CLASSIFIERS


for classifier in CLASSIFIERS:
    print(classifier.__name__)
    model, acc, prec, rec,eer, cm_display, fEER = model_creation(classifier, ['bonafide', 'spoof'])
    for i in range(0):
        tmp_model, tmp_acc, tmp_prec, tmp_rec, tmp_eer, tmp_cm_display, tmp_fEER = model_creation(classifier, ['bonafide', 'spoof'])
        if tmp_acc > acc:
            model = tmp_model
            acc = tmp_acc
            prec = tmp_prec
            rec = tmp_rec
            eer = tmp_eer
            cm_display = tmp_cm_display
            fEER = tmp_fEER
    cm_display.plot()

    #this script will save confusion matrix, frr and far curves, metrics in a yaml and the model
    # 4 files for every model

    plt.savefig('./models/' + classifier.__name__ + '_conf_matrix.png')
    fEER.savefig('./models/' + classifier.__name__ + '_frr_frr_curve.png')
    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "EER": eer
    }
    with open('./models/' + classifier.__name__ + '_metrics.yml', 'w') as f: 
        for key, value in metrics.items(): 
            f.write('%s: %s\n' % (key, value))

    pickle.dump(model, open('./models/' + classifier.__name__ + '.pkl', 'wb'))




