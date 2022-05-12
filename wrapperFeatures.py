#librerie per le features
from spafe.features.bfcc import bfcc
from spafe.features.lfcc import lfcc
from spafe.features.lpc import lpc, lpcc
from spafe.features.mfcc import mfcc, imfcc
from spafe.features.msrcc import msrcc
from spafe.features.ngcc import ngcc
from spafe.features.pncc import pncc
from spafe.features.psrcc import psrcc
from spafe.features.rplp import rplp, plp
from spafe.fbanks import mel_fbanks , bark_fbanks , gammatone_fbanks
from spafe.features.spfeats import extract_feats

import statistics

def computeFeatures(fileAudio,samplerate):

    output = []
    #definire valori di input ulteriori per le seguenti funzioni

    #*************************************************

    output.append(bfcc(fileAudio,samplerate).mean())

    output.append(lfcc(fileAudio,samplerate).mean())

    output.append(lpc(fileAudio,samplerate).mean())

    output.append(lpcc(fileAudio,samplerate).mean())

    output.append(mfcc(fileAudio,samplerate).mean())

    output.append(imfcc(fileAudio,samplerate).mean())
    
    output.append(msrcc(fileAudio,samplerate).mean())

    output.append(ngcc(fileAudio,samplerate).mean())


    """
    # da errori sui file flac
    output.append(pncc().mean()) #buggata??
    """
    output.append(psrcc(fileAudio,samplerate).mean())

    output.append(plp(fileAudio,samplerate).mean())

    output.append(rplp(fileAudio,samplerate).mean())

    #definire valori di input per metodi di filter_banks
    output.append(mel_fbanks.mel_filter_banks(fs = samplerate).mean())

    output.append(bark_fbanks.bark_filter_banks(fs = samplerate).mean())

    output.append(gammatone_fbanks.gammatone_filter_banks(fs = samplerate).mean())

    d = extract_feats(fileAudio,samplerate)
    l = list(d.values())

    #il primo elemento è la durata del file audio, non è un dato che aiuta nella classificazione
    l.pop(0)


    #convertiamo ndarray in float (con la media)
    #convertiamo tuple sempre in una media
    #liste le convertiamo in una media (se la lista è vuota mettiamo 0)
    #i numeri complessi li convertiamo in float tramite abs()
    for i in range(len(l)):
        if type(l[i]).__name__ == "ndarray":
            l[i] = l[i].mean()
        elif type(l[i]).__name__ == "tuple":
            l[i] = l[i][0].mean()
        elif type(l[i]).__name__ == "list":
            if len(l[i]) > 0:
                l[i] = statistics.mean(l[i])
            else:
                l[i] = 0
        elif type(l[i]).__name__ == "complex128":
            l[i] = abs(l[i])

    output = output + l
    return output
