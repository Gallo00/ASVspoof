import csv
import tarfile 
import soundfile as sf 
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import scipy

from wrapperFeatures import computeFeatures

from getNamesFeatures import getNamesFeatures

from constants import FEATURES

# open txt and extract
tLabels = tarfile.open("set_tesi\DF-keys-stage-1.tar.gz","r:gz")
txt = tLabels.extractfile(tLabels.getmember('keys/CM/trial_metadata.txt'))


# read lines; each line has <filename> - <label>
listLines = txt.readlines()


tPart00 = tarfile.open("set_tesi\ASVspoof2021_DF_eval_part00.tar.gz","r:gz")
arrFilesPart00 = tPart00.getmembers()
print("getMembers Part00 completed")

# element 0: ASVspoof2021_DF_eval/ASVspoof2021.DF.cm.eval.trl.txt
# element 1: ASVspoof2021_DF_eval/LICENSE.DF.txt
# element 2: ASVspoof2021_DF_eval/README.DF.txt
# remaining elements: flac files

# remove first 3 elements of arrFilesPart00
arrFilesPart00.pop(0)
arrFilesPart00.pop(0)
arrFilesPart00.pop(0)

# in the other tar.gz we don't have to do that

arr = arrFilesPart00[0:10]


# create csv and set the header 
with open('prova.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["file","label"] + FEATURES)
    file.close()

file = open('prova.csv','a',newline='')
# disable warnings
#https://stackoverflow.com/questions/14463277/how-to-disable-python-warnings
for i in tqdm(range(len(arr))):

    l = listLines[i]
    arrRow = l.decode().split(" ") 
    rowF = []
    rowF.append(arrRow[1] + ".flac") # we are going to use elements from position 1 and 5 (1: file , 5: label)
    rowF.append(arrRow[5])

    # extract and read file
    fileAudio = tPart00.extractfile(arr[i].name)
    fileAudioR , samplerate = sf.read(fileAudio)

    lF = computeFeatures(fileAudioR,samplerate)

    # creation of new line 
    nomeFile = arr[i].name.split("/")[-1] 
    r = rowF + lF # file_name.flac - label - feature1 - feature2 ... featureN

    # append new line to csv
    writer = csv.writer(file)
    writer.writerow(r)

# enable warnings
#https://stackoverflow.com/questions/29784889/python-how-to-enable-all-warnings
file.close()

"""
#in part01, 02 e 03 non ci sono questi elementi
tPart01 = tarfile.open("set_tesi\ASVspoof2021_DF_eval_part01.tar.gz","r:gz")
arrFilesPart01 = tPart01.getmembers()
print("getMembers Part01 completata")

tPart02 = tarfile.open("set_tesi\ASVspoof2021_DF_eval_part02.tar.gz","r:gz")
arrFilesPart02 = tPart02.getmembers()
print("getMembers Part02 completata")

tPart03 = tarfile.open("set_tesi\ASVspoof2021_DF_eval_part03.tar.gz","r:gz")
arrFilesPart03 = tPart03.getmembers()
print("getMembers Part03 completata")



txt.close()
tLabels.close()
tPart00.close()
tPart01.close()
tPart02.close()
tPart03.close()

"""

"""
str = "- DF_E_2000011 - - - spoof - progress"
arr = str.split(" ")
print(arr)
arr.pop()
print(arr)
"""