import csv 
import soundfile as sf 
import os
from multiprocessing import Pool
from tqdm import tqdm

from wrapper_features import compute_features
import mutagen

from constants import FEATURES

import warnings


def writeCSV(dataset: str, pack):
    warnings.filterwarnings("ignore")
    with open(dataset, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["file","label"] + FEATURES)
        file.close()

    file = open(dataset,'a+',newline='')
    writer = csv.writer(file)
    print("writing lines from " + pack[3] + " into the csv file...")
    for i in tqdm(range(pack[1])): 
        arr_row = (pack[2])[i].split(" ") 
        row_file = []
        row_file.append(arr_row[1] + ".flac") # we are going to use elements from position 1 and 5 (1: file , 5: label)
        row_file.append(arr_row[5])

        file_audio = pack[0][i]
        file_audioR , samplerate = sf.read(pack[4] + file_audio)

        list_features = compute_features(file_audioR,samplerate)

        f = mutagen.File(pack[4] + file_audio)
        bitrate = f.info.bitrate

        list_features.append(bitrate)

        # creation of new line 
        # file_name.flac - label - feature1 - feature2 ... featureN
        # append new line to csv
        writer.writerow(row_file + list_features)
    warnings.simplefilter('always')
    file.close()

def main():
    # open txt
    txt = open("set_tesi/DF-keys-stage-1/keys/CM/trial_metadata.txt",mode='r')


    # read lines; each line has <filename> - <label>
    list_lines = txt.readlines()


    # get name of all files 
    arr_files_part00 = os.listdir("set_tesi/ASVspoof2021_DF_eval_part00/ASVspoof2021_DF_eval/flac")
    arr_files_part01 = os.listdir("set_tesi/ASVspoof2021_DF_eval_part01/ASVspoof2021_DF_eval/flac")
    arr_files_part02 = os.listdir("set_tesi/ASVspoof2021_DF_eval_part02/ASVspoof2021_DF_eval/flac")
    arr_files_part03 = os.listdir("set_tesi/ASVspoof2021_DF_eval_part03/ASVspoof2021_DF_eval/flac")



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


    packs = [[arr_files_part00, len00, lines00, "Part00","set_tesi/ASVspoof2021_DF_eval_part00/ASVspoof2021_DF_eval/flac/"],
            [arr_files_part01, len01, lines01, "Part01","set_tesi/ASVspoof2021_DF_eval_part01/ASVspoof2021_DF_eval/flac/"],
            [arr_files_part02, len02, lines02, "Part02","set_tesi/ASVspoof2021_DF_eval_part02/ASVspoof2021_DF_eval/flac/"],
            [arr_files_part03, len03, lines03, "Part03","set_tesi/ASVspoof2021_DF_eval_part03/ASVspoof2021_DF_eval/flac/"],]
    #0: array of Files
    #1: length of the array
    #2: lines in txt file
    #3: name of pack
    #4: path

    # disable warnings
    #https://stackoverflow.com/questions/14463277/how-to-disable-python-warnings


    cpus = os.cpu_count()
    print(cpus)
    
    
    with Pool(cpus) as pool:
        pool.starmap(writeCSV, [('datasetPart00.csv',packs[0]),
                            ('datasetPart01.csv',packs[1]),
                            ('datasetPart02.csv',packs[2]),
                            ('datasetPart03.csv',packs[3])])
    #print(result)


    # enable warnings
    #https://stackoverflow.com/questions/29784889/python-how-to-enable-all-warnings
    txt.close()

if __name__ == "__main__":
    main()

