import pandas as pd
from constants import FEATURES, SAMPLE_ROWS_STATS
from tqdm import tqdm 
from create_fig import clear_plt, save_fig_single_plot, save_fig_double_plot, name_axes
dataframe1 = pd.read_csv('datasetPart00l.csv')
dataframe2 = pd.read_csv('datasetPart01l.csv')
dataframe3 = pd.read_csv('datasetPart02l.csv')
dataframe4 = pd.read_csv('datasetPart03l.csv')

dataframe = pd.concat([dataframe1,dataframe2,dataframe3,dataframe4],axis=0)

dataframe.pop('spectral_variance')
#dataframe.pop('label')
# spectral_variance could slow down the script without complete the calc


# split dataframe
dataframe_bonafide = dataframe.loc[dataframe['label'] == 'bonafide']
dataframe_spoof = dataframe.loc[dataframe['label'] == 'spoof']

#df1_elements = df1.sample(n=4)
SAMPLES = 251
dataframe_bonafide = dataframe_bonafide.sample(SAMPLES)
dataframe_spoof = dataframe_spoof.sample(SAMPLES)

name_axes()

methods = ['freedman_diaconis','knuth']

#methods = ['freedman_diaconis']
#methods = ['knuth']
for method in methods:
    print(method)
    for feat in tqdm(FEATURES):
        try:
            y_lims, x_lims = save_fig_double_plot(feat, dataframe_bonafide, dataframe_spoof, method)
            clear_plt()

            curve_deepfake, lsize_deepfake = save_fig_single_plot(feat,dataframe_spoof,method,'deepfake', y_lims, x_lims)
            clear_plt()

            curve_bonafide, lsize_bonafide = save_fig_single_plot(feat,dataframe_bonafide,method,'bonafide', y_lims, x_lims)
            clear_plt()
        except:
            print("can't calculate for " + feat)







