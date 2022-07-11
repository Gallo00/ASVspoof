import matplotlib.pyplot as plt
import numpy as np
from numpy import void
import pandas as pd
from create_curve import create_curve
from typing import Union

def clear_plt() -> void:
    plt.clf()
    plt.cla()
    plt.close()

def save_fig_single_plot(feat: str, dataframe: pd.DataFrame, method: str, folder: str) -> Union[np.ndarray,int]:
    color = 'red'
    if(folder == 'bonafide'):
        color = 'blue'

    plt.title(feat)

    col = dataframe[feat].tolist()
    curve, lsize = create_curve(col, method=method) 
    plt.plot(curve, color=color)

    plt.savefig('img_feat_' + method + '/' + folder + '/' + feat + '.png')
    return curve, lsize

def save_fig_double_plot(feat: str, lsize_deepfake: int, curve_deepfake: np.ndarray,
 lsize_bonafide: int, curve_bonafide: np.ndarray, method: str) -> void:
    plt.title(feat)

    plt.plot(curve_deepfake, color='red')
    plt.plot(curve_bonafide, color='blue')

    leg_str_deepfake = 'deepfake (' + str(lsize_deepfake) + ')'
    leg_str_bonafide = 'bonafide (' + str(lsize_bonafide) + ')'
    plt.legend([leg_str_deepfake,leg_str_bonafide])

    plt.savefig('img_feat_' + method + '/bonafide_deepfake/' + feat + '.png')