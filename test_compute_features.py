from wrapper_features import computeFeatures

from spafe.features.spfeats import extract_feats
import soundfile as sf 


fileAudioLetto , samplerate = sf.read("set_tesi\ASVspoof2021_DF_eval_part00\DF_E_2000080.flac")
#print(fileAudioLetto)
#print(samplerate)

arr = computeFeatures(fileAudioLetto,samplerate)

print(len(arr))
print(arr)

