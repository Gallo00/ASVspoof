import csv
import tarfile 
import soundfile as sf 
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import scipy

from wrapperFeatures import compute_features

from constants import FEATURES

import warnings

# open txt and extract
tLabels = tarfile.open("set_tesi\DF-keys-stage-1.tar.gz","r:gz")
txt = tLabels.extractfile(tLabels.getmember('keys/CM/trial_metadata.txt'))


# read lines; each line has <filename> - <label>
list_lines = txt.readlines()


tPart00 = tarfile.open("set_tesi\ASVspoof2021_DF_eval_part00.tar.gz","r:gz")
arr_files_part00 = tPart00.getmembers()
print("getmembers Part00 completed")

tPart01 = tarfile.open("set_tesi\ASVspoof2021_DF_eval_part01.tar.gz","r:gz")
arr_files_part01 = tPart01.getmembers()
print("getmembers Part01 completed")

tPart02 = tarfile.open("set_tesi\ASVspoof2021_DF_eval_part02.tar.gz","r:gz")
arr_files_part02 = tPart02.getmembers()
print("getmembers Part02 completed")

tPart03 = tarfile.open("set_tesi\ASVspoof2021_DF_eval_part03.tar.gz","r:gz")
arr_files_part03 = tPart03.getmembers()
print("getmembers Part03 completed")


# NOTES FOR PART00
# element 0: ASVspoof2021_DF_eval/ASVspoof2021.DF.cm.eval.trl.txt
# element 1: ASVspoof2021_DF_eval/LICENSE.DF.txt
# element 2: ASVspoof2021_DF_eval/README.DF.txt
# remaining elements: flac files

# remove first 3 elements of arr_files_part00
arr_files_part00.pop(0)
arr_files_part00.pop(0)
arr_files_part00.pop(0)

# in the other tar.gz we don't have to do that

#create a list that contains all files
arr_files_set = arr_files_part00 + arr_files_part01 + arr_files_part02 + arr_files_part03

len00 = len(arr_files_part00) # 152955
len01 = len(arr_files_part01) # 152958
len02 = len(arr_files_part02) # 152958
len03 = len(arr_files_part03) # 152958
# len arr_files_set: 611829

#slice list_lines in 4 sublists (the first has len00 elements)
lines00 = list_lines[0:len00]
lines01 = list_lines[len00:(len00 + len01)]
lines02 = list_lines[(len00 + len01):(len00 + len01 + len02)]
lines03 = list_lines[(len00 + len01 + len02):(len00 + len01 + len02 + len03)]


packs = [[tPart00, arr_files_part00, len00, lines00, "Part00"],
         [tPart01, arr_files_part01, len01, lines01, "Part01"],
         [tPart02, arr_files_part02, len02, lines02, "Part02"],
         [tPart03, arr_files_part03, len03, lines03, "Part03"],]
#0: tarFile obj
#1: array of Files
#2: length of the array
#3: lines in txt file
#4: name of file


# create csv and set the header 
with open('prova.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["file","label"] + FEATURES)
    file.close()

file = open('prova.csv','a',newline='')
writer = csv.writer(file)


# disable warnings
#https://stackoverflow.com/questions/14463277/how-to-disable-python-warnings

warnings.filterwarnings("ignore")

for pack in packs:
    print("writing lines from " + pack[4] + " into the csv file...")
    for i in tqdm(range(pack[2])):
        arr_row = (pack[3])[i].decode().split(" ") 
        row_file = []
        row_file.append(arr_row[1] + ".flac") # we are going to use elements from position 1 and 5 (1: file , 5: label)
        row_file.append(arr_row[5])

        # extract and read file
        file_audio = pack[0].extractfile((pack[1])[i].name)
        file_audioR , samplerate = sf.read(file_audio)

        list_features = compute_features(file_audioR,samplerate)

        # creation of new line 
        #r = rowF + lF # file_name.flac - label - feature1 - feature2 ... featureN
        # append new line to csv
        writer.writerow(row_file + list_features)

# enable warnings
#https://stackoverflow.com/questions/29784889/python-how-to-enable-all-warnings

warnings.simplefilter('always')

file.close()

txt.close()
tLabels.close()
tPart00.close()
tPart01.close()
tPart02.close()
tPart03.close()


"""
for i in tqdm(range(len(arrFilesSet))):

    #l = list_lines[i]
    arrRow = list_lines[i].decode().split(" ") 
    rowF = []
    rowF.append(arrRow[1] + ".flac") # we are going to use elements from position 1 and 5 (1: file , 5: label)
    rowF.append(arrRow[5])

    # extract and read file
    tPart = None
    if(i < len00): tPart = tPart00
    elif (i < len01): tPart = tPart01
    elif (i < len02): tPart = tPart02
    else: tPart = tPart03


    file_audio = tPart.extractfile(arrFilesSet[i].name)
    file_audioR , samplerate = sf.read(file_audio)

    lF = computeFeatures(file_audioR,samplerate)

    # creation of new line 
    #r = rowF + lF # file_name.flac - label - feature1 - feature2 ... featureN
    # append new line to csv
    writer.writerow(rowF + lF)

"""
