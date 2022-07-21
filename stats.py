import pandas as pd
from constants import FEATURES, SAMPLE_ROWS_STATS
from tqdm import tqdm 
from create_fig import clear_plt, save_fig_single_plot, save_fig_double_plot, name_axes
dataframe = pd.read_csv('data_20k.csv')

dataframe.pop('spectral_variance')
dataframe.pop('label')
# spectral_variance could slow down the script without complete the calc

txt = open("set_tesi/DF-keys-stage-1/keys/CM/trial_metadata.txt",mode='r')

# read lines; each line has <filename> - <label>
list_lines = txt.readlines()

dict_lines = {}
for i in range(len(list_lines)):
    dict_lines[list_lines[i].split(" ")[1] + ".flac"] =  list_lines[i].split(" ")[5]

#print(dict_lines)

#print(dataframe.head())

# split dataframe
#dataframe_bonafide = dataframe.loc[dataframe['label'] == 'bonafide']
#dataframe_spoof = dataframe.loc[dataframe['label'] == 'spoof']

dataframe_bonafide = pd.DataFrame(columns=["AUDIO_FILE_NAME"] + FEATURES)
dataframe_spoof = pd.DataFrame(columns=["AUDIO_FILE_NAME"] + FEATURES)

for k,v in tqdm(dict_lines.items()):
    if v == "spoof":
        new_row = dataframe.loc[dataframe['AUDIO_FILE_NAME'] == k]
        dataframe_spoof = pd.concat([dataframe_spoof,new_row])
    if v == "bonafide":
        new_row = dataframe.loc[dataframe['AUDIO_FILE_NAME'] == k]
        dataframe_bonafide = pd.concat([dataframe_bonafide,new_row])

#df1_elements = df1.sample(n=4)
#dataframe_bonafide = dataframe_bonafide.sample(1000)
#dataframe_spoof = dataframe_spoof.sample(1000)

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







