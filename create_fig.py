import matplotlib.pyplot as plt
from numpy import void
import pandas as pd
from create_curve import create_curve
from typing import Union

def clear_plt() -> void:
    plt.clf()
    plt.cla()
    plt.close()

def save_fig_single_plot(feat: str, dataframe, method: str, folder: str):
    color = 'red'
    if(folder == 'bonafide'):
        color = 'blue'

    plt.title(feat)

    col = dataframe[feat].tolist()
    curve, lsize = create_curve(col, method=method) 
    plt.plot(curve, color=color)

    plt.savefig('img_feat_' + method + '/' + folder + '/' + feat + '.png')
    return curve, lsize

def save_fig_double_plot(feat, lsize_deepfake, curve_deepfake,
 lsize_bonafide, curve_bonafide, method):
    plt.title(feat)

    plt.plot(curve_deepfake, color='red')
    plt.plot(curve_bonafide, color='blue')

    leg_str_deepfake = 'deepfake (' + str(lsize_deepfake) + ')'
    leg_str_bonafide = 'bonafide (' + str(lsize_bonafide) + ')'
    plt.legend([leg_str_deepfake,leg_str_bonafide])

    plt.savefig('img_feat_' + method + '/bonafide_deepfake/' + feat + '.png')