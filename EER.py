#https://stackoverflow.com/questions/30503766/how-can-i-calculate-the-failed-acceptance-rate-and-false-recognition-rate
"""
FAR = FP / (FP + TN)
FRR = FN / (FN + TP)
Where:

FAR: False Acceptance Rate
FRR: False Rejection Rate
TP: True Positive
FP: False Positive
TN: True Negative
FN: False Negative
"""
import numpy as np
import matplotlib.pyplot as plt

def curve_frr_far(
    targets, #list of 0 and 1
    genuine_probabilities,
    genuine_label = 0 # 0 or 1
):
    sort_idxs = genuine_probabilities.argsort()
    targets = targets[sort_idxs]
    genuine_probabilities = genuine_probabilities[sort_idxs]
    
    
    far = []
    frr = []
    thresholds = []

    len_genuines = (targets == genuine_label).sum() 
    len_spoofs = (targets != genuine_label).sum()
    
    len_targets = len(targets)
    th = 0.0
    fa = len_spoofs
    fr = 0

    thresholds.append(th)
    far.append(fa/len_spoofs)
    frr.append(fr/len_genuines)
        
    for index in range(len_targets):
        genuine_p = genuine_probabilities[index]
        label = targets[index]

        if label == genuine_label:
            fr += 1
        else:
            fa -= 1

        if genuine_p != th:
            th = genuine_p
            far.append(fa/len_spoofs)
            frr.append(fr/len_genuines)
            thresholds.append(th)

    if th != 1.0:
        th = 1.0
        fa = 0
        fr = len_genuines
        far.append(fa/len_spoofs)
        frr.append(fr/len_genuines)
        thresholds.append(th)
    
    return thresholds, frr, far


def eer(threshold, frr, far):
    for i in range(len(frr)):
        if np.isnan(frr[i]):
            frr[i] = 0.0
        if np.isnan(far[i]):
            far[i] = 0.0
        if np.isnan(threshold[i]):
            threshold[i] = 0.0
        frr[i] = int(frr[i]*100)
        far[i] = int(far[i]*100)
        threshold[i] = int(threshold[i]*100)

    for i  in range(len(frr)):
        a = frr[i]
        b = far[i]
        if a == b:
            EER = a
            break
    fig, ax = plt.subplots()

    ax.plot(threshold, far, 'r--', label='FAR')
    ax.plot(threshold, frr, 'g--', label='FRR')
    plt.xlabel('Threshold')
    plt.plot(15,EER,'ro', label='EER') 


    legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')

    # Put a nicer background color on the legend.
    legend.get_frame().set_facecolor('C0')

    plt.show()
    return EER