from model_creation import model_creation
import pickle
from constants import CLASSIFIERS


for classifier in CLASSIFIERS:
    print(classifier.__name__)
    model, acc = model_creation(classifier)
    for i in range(2):
        if acc >= 0.95:
            break
        else:
            tmp_model, tmp_acc = model_creation(classifier)
            if tmp_acc > acc:
                model = tmp_model
                acc = tmp_acc
    pickle.dump(model, open('./models/' + classifier.__name__ + '.pkl', 'wb'))



