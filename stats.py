import matplotlib.pyplot as plt
import pandas as pd
from create_curve import create_curve
from constants import FEATURES
from tqdm import tqdm 
from create_fig import clear_plt, save_fig_single_plot, save_fig_double_plot
dataframe = pd.read_csv('temp_ds.csv')
#dataframe = pd.read_csv('dataset.csv')

dataframe.pop('spectral_variance')

#print(dataframe.head())

#split dataframe
dataframe_bonafide = dataframe.loc[dataframe['label'] == 'bonafide']
dataframe_spoof = dataframe.loc[dataframe['label'] == 'spoof']

#df1_elements = df1.sample(n=4)
dataframe_bonafide = dataframe_bonafide.sample(300)
dataframe_spoof = dataframe_spoof.sample(300)

#extract a column, convert into a list and calculate number of bins

"""
col = dataframe_spoof['psrcc'].tolist()
curve, lsize = create_curve(col) 

plt.plot(curve, color='red')

col = dataframe_bonafide['psrcc'].tolist()
curve, lsize = create_curve(col) 
plt.plot(curve, color='blue')
plt.title('psrcc')

leg_str_deepfake = 'deepfake (' + str(lsize) + ')'
leg_str_bonafide = 'bonafide (' + str(lsize) + ')'
plt.legend([leg_str_deepfake,leg_str_bonafide])

plt.savefig('img_feat/psrcc.png')
"""

methods = ['freedman_diaconis','knuth']
#methods = ['freedman_diaconis']
#methods = ['knuth']
for method in methods:
    print(method)
    for feat in tqdm(FEATURES):
        try:
            curve_deepfake, lsize_deepfake = save_fig_single_plot(feat,dataframe_spoof,method,folder='deepfake')
            clear_plt()

            curve_bonafide, lsize_bonafide = save_fig_single_plot(feat,dataframe_bonafide,method,folder='bonafide')
            clear_plt()
            
            save_fig_double_plot(feat,lsize_deepfake,curve_deepfake,lsize_bonafide,curve_bonafide,method)
            clear_plt()
        except:
            print("can't calculate for " + feat)







