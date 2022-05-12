import csv
import tarfile #per i file tar.gz
import soundfile as sf #per i file flac
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import scipy

from wrapperFeatures import computeFeatures

from getNamesFeatures import getNamesFeatures

#apriamo file txt con tarfile e estraimolo
tLabels = tarfile.open("set_tesi\DF-keys-stage-1.tar.gz","r:gz")
txt = tLabels.extractfile(tLabels.getmember('keys/CM/trial_metadata.txt'))


#leggiamo le righe del file con le labels
listLines = txt.readlines()


#apriamo in successione i 4 tar.gz
tPart00 = tarfile.open("set_tesi\ASVspoof2021_DF_eval_part00.tar.gz","r:gz")
arrFilesPart00 = tPart00.getmembers()
print("getMembers Part00 completata")

#elemento 0: ASVspoof2021_DF_eval/ASVspoof2021.DF.cm.eval.trl.txt
#elemento 1: ASVspoof2021_DF_eval/LICENSE.DF.txt
#elemento 2: ASVspoof2021_DF_eval/README.DF.txt
#restanti elementi: file flac

#rimuoviamo i primi 3 elementi dall'array
arrFilesPart00.pop(0)
arrFilesPart00.pop(0)
arrFilesPart00.pop(0)

#negli altri tar.gz non ci sono questi elementi

arr = arrFilesPart00[0:10]

#OTTENIAMO I NOMI DELLE FEATURES
arrNames = getNamesFeatures()


#aprire/creare file csv e aggiungere l'header
with open('prova.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(getNamesFeatures())
    file.close()

file = open('prova.csv','a',newline='')

#print(dfLabels[dfLabels["fileFlac"]=="DF_E_2000013"].loc[1]["label"]) #come ottenere la label del fileFlac DF_E_2000013
#disattivare warnings
#https://stackoverflow.com/questions/14463277/how-to-disable-python-warnings
for i in tqdm(range(len(arr))):

    l = listLines[i]
    arrRow = l.decode().split(" ") #ogni linea viene splittata in tokens
    rowF = []
    rowF.append(arrRow[1] + ".flac") #conserviamo i token 1 e 5, sono rispettivamente il nome del file(senza .flac) e l'etichetta
    rowF.append(arrRow[5])

    #estrazione e lettura file
    fileAudio = tPart00.extractfile(arr[i].name)
    fileAudioLetto , samplerate = sf.read(fileAudio)

    #calcolo features
    lF = computeFeatures(fileAudioLetto,samplerate)
    #creazione della nuova riga e push nel dataframe
    nomeFile = arr[i].name.split("/")[-1] #ci da il nome del file (con .flac)

    r = rowF + lF # nome_file.flac - label - feature1 - feature2 ... featureN

    writer = csv.writer(file)
    writer.writerow(r)
#riattivare warnings
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