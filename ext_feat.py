from wrapper_features import compute_features
import mutagen
import soundfile as sf 

file_audio_spoof , samplerate_spoof = sf.read("set_tesi/ASVspoof2021_DF_eval_part00/ASVspoof2021_DF_eval/flac/DF_E_2000011.flac")

file_audio_bonafide , samplerate_bonafide = sf.read("set_tesi/ASVspoof2021_DF_eval_part00/ASVspoof2021_DF_eval/flac/DF_E_2000053.flac")

list_features_spoof = compute_features(file_audio_spoof,samplerate_spoof)
f = mutagen.File("set_tesi/ASVspoof2021_DF_eval_part00/ASVspoof2021_DF_eval/flac/DF_E_2000011.flac")
bitrate = f.info.bitrate
list_features_spoof.append(bitrate)

print(list_features_spoof)
print("-----------------------------------------------------------------")

list_features_bonafide = compute_features(file_audio_bonafide,samplerate_bonafide)
f = mutagen.File("set_tesi/ASVspoof2021_DF_eval_part00/ASVspoof2021_DF_eval/flac/DF_E_2000053.flac")
bitrate = f.info.bitrate
list_features_bonafide.append(bitrate)

print(list_features_bonafide)

