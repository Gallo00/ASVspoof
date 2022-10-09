# ASVspoof
Progetto software per tesi triennale, soluzione della challenge ASVspoof DF

### Usage
To start features_extraction: python features_extraction.py <br>
It will create the file 'dataset.csv'. <br>
Previously you have to create the directory "set_tesi" (in the same directory of features_extraction.py)

set_tesi has to have this structure

``` 
set_tesi
|
|-ASVspoof2021_DF_eval_part00
   |
   |-ASVspoof2021_DF_eval
      |
      |-flac
        |
        |-all flac files-
   |-ASVspoof2021.DF.cm.eval.trl.txt
   |
   |-LICENSE.DF.txt
   |
   |-README.DF.txt
|
|-ASVspoof2021_DF_eval_part01
   |
   |-ASVspoof2021_DF_eval
      |
      |-flac
        |
        |-all flac files-
|
|-ASVspoof2021_DF_eval_part02
   |
   |-ASVspoof2021_DF_eval
      |
      |-flac
        |
        |-all flac files-
|
|-ASVspoof2021_DF_eval_part03
   |
   |-ASVspoof2021_DF_eval
      |
      |-flac
        |
        |-all flac files-
|        
|-DF-keys-stage-1
  |
  |-keys
    |
    |-CM
      |
      |-trial_metadata.txt
```
<br>
After features extraction we can launch python stats.py <br>
It will create and save some charts into the directories img_feat_knuth and img_feat_freedman_diaconis. <br>
The charts represent the behavior of various features for both labels (bonafide and spoof) <br>

The script create_models.py will train and save various models. <br>
The script naive_classifier_th.py will create a naive model based on research of an good threshold of feature bit_rate to classify audio files.
The script naive_classifier_mean.py is similar to the classifier based on threshold, but this one is based on mean of bit_rate.
