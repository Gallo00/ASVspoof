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

def save_fig_single_plot(feat: str, dataframe: pd.DataFrame, method: str, folder: str,
 y_lims: tuple, x_lims: tuple) -> Union[np.ndarray,int]:
    color = 'red'
    if(folder == 'bonafide'):
        color = 'blue'

    plt.title(feat)

    col = dataframe[feat].tolist()
    curve,bincenters, lsize = create_curve(col, method=method) 
    plt.plot(curve,bincenters, color=color)
    plt.gca().set_ylim(y_lims)
    plt.gca().set_xlim(x_lims)

    plt.savefig('img_feat_' + method + '/' + folder + '/' + feat + '.png')
    return curve, lsize

def save_fig_double_plot(feat: str, dataframe_bonafide: pd.DataFrame,
 dataframe_deepfake: pd.DataFrame, method: str) -> Union[tuple, tuple]:
    plt.title(feat)

    col_bonafide = dataframe_bonafide[feat].tolist()
    curve_bonafide,bincenters_bonafide, lsize_bonafide = create_curve(col_bonafide, method=method)

    col_deepfake = dataframe_deepfake[feat].tolist()
    curve_deepfake,bincenters_deepfake, lsize_deepfake = create_curve(col_deepfake, method=method)


    plt.plot(bincenters_deepfake,curve_deepfake, color='red')
    plt.plot(bincenters_bonafide,curve_bonafide, color='blue')

    leg_str_deepfake = 'deepfake (' + str(lsize_deepfake) + ')'
    leg_str_bonafide = 'bonafide (' + str(lsize_bonafide) + ')'
    plt.legend([leg_str_deepfake,leg_str_bonafide])

    y_lims = plt.gca().get_ylim()
    x_lims = plt.gca().get_xlim()
    plt.savefig('img_feat_' + method + '/bonafide_deepfake/' + feat + '.png')

    return y_lims, x_lims