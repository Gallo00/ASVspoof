def compute_eer(TP: int, FN: int, FP: int, TN: int) -> float:
    FAR = FP/(TP + FP + TN + FN)
    FRR = FN/(TP + FP + TN + FN)
    EER = (FAR + FRR) / 2
    return EER

def compute_accperclass(TP: int, FN: int, FP: int, TN: int) -> float:
    TPR = TP/(TP+FN)
    FPR = FP/(FP+TN)
    ACC_CLASS = (TPR + FPR) / 2
    return ACC_CLASS
    