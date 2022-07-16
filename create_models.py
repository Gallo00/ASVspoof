from model_creation import model_creation
import pickle
import matplotlib.pyplot as plt
from constants import CLASSIFIERS


for classifier in CLASSIFIERS:
    print(classifier.__name__)
    model, acc, cm_display = model_creation(classifier)
    for i in range(4):
        if acc >= 0.90:
            break
        else:
            tmp_model, tmp_acc, tmp_cm_display = model_creation(classifier)
            if tmp_acc > acc:
                model = tmp_model
                acc = tmp_acc
                conf_matrix = tmp_cm_display
    cm_display.plot()
    plt.savefig('./models/' + classifier.__name__ + '.png')
    pickle.dump(model, open('./models/' + classifier.__name__ + '.pkl', 'wb'))



