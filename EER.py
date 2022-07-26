# https://stackoverflow.com/questions/30503766/how-can-i-calculate-the-failed-acceptance-rate-and-false-recognition-rate
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
import sys

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


def eer(threshold: list, frr: list, far: list, show_graph = False) -> float:
    
    for i in range(len(frr)):
        if np.isnan(frr[i]):
            frr[i] = 0.0

        if np.isnan(far[i]):
            far[i] = 0.0

        if np.isnan(threshold[i]):
            threshold[i] = 0.0

        dec = 10
        frr[i] = round(frr[i] * 100, dec)
        far[i] = round(far[i] * 100, dec)
        threshold[i] = round(threshold[i] * 100, dec)

    eer_y = sys.float_info.max
    eer_x = 0

    for i  in range(len(frr)):
        _frr = frr[i]
        _far = far[i]

        candidate_y = abs(_frr - _far)
        if eer_y > candidate_y:
            eer_y = candidate_y
            eer_x = threshold[i]

    if show_graph:
        _, ax = plt.subplots()

        ax.plot(threshold, far, 'r--', label='FAR')
        ax.plot(threshold, frr, 'g--', label='FRR')
        plt.xlabel('Threshold')
        plt.plot(eer_x, eer_y, 'ro', label='EER') 

        legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')

        # Put a nicer background color on the legend.
        legend.get_frame().set_facecolor('C0')

        plt.show()

    return eer_y
