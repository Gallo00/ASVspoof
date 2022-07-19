import pandas as pd
from constants import FEATURES, SAMPLE_ROWS_STATS
from tqdm import tqdm 
from create_fig import clear_plt, save_fig_single_plot, save_fig_double_plot
dataframe = pd.read_csv('dataset.csv')

dataframe.pop('spectral_variance')
# spectral_variance could slow down the script without complete the calc

#print(dataframe.head())

# split dataframe
dataframe_bonafide = dataframe.loc[dataframe['label'] == 'bonafide']
dataframe_spoof = dataframe.loc[dataframe['label'] == 'spoof']

#df1_elements = df1.sample(n=4)
dataframe_bonafide = dataframe_bonafide.sample(SAMPLE_ROWS_STATS)
dataframe_spoof = dataframe_spoof.sample(SAMPLE_ROWS_STATS)


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







