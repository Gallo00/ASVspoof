# Fake Audio Detector
Questo progetto si propone l'obiettivo di costruire uno strumento in grado di suddividere file audio in 2 categorie: file audio con voci sintetiche e file audio con voci reali. <br>
Il tool è costruito tramite modelli di machine learning, per fare ciò ci si serve della libreria sklearn di Python che offre modelli già pronti per l'allenamento, il testing e l'utilizzo vero e proprio.
Il dataset utilizzato è reperibile <a href="https://zenodo.org/record/4835108">qui</a>, pesa approssimativamente 32GB.
Si tratta del dataset offerto da <a href="https://www.asvspoof.org/">ASVspoof</a> che ogni 2 anni propone una challenge che comprende anche la creazione di un modello che possa contraddistinguere audio con voci reali da audio con voci sintetiche. 

### Estrazione delle features
Come step preliminare per la creazione del modello è stato necessario lavorare ad un'estrazione delle features in quanto il dataset contiene dei veri e propri file audio. 
E' necessario per ogni file ottenerne una rappresentazione numerica.
Per questo lavoro lo strumento utilizzato in python è la libreria <a href="https://github.com/SuperKogito/spafe">spafe</a> (versione 0.1.2).
Tramite delle semplici chiamate a funzioni si sono estratte le seguenti features:
bfcc, lfcc, lpc,lpcc,mfcc, imfcc, msrcc, ngcc, psrcc, plp, rplp, mel_filter_banks, bark_filter_banks, gammatone_filter_banks, spectrum, mean_frequency, peak_frequency, frequencies_std, amplitudes_cum_sum,mode_frequency, median_frequency, frequencies_q25, frequencies_q75, iqr, freqs_skewness, freqs_kurtosis,spectral_entropy, spectral_flatness, spectral_centroid, spectral_bandwidth, spectral_spread, spectral_rolloff,energy,rms, zcr, spectral_mean, spectral_rms, spectral_std, spectral_variance, meanfun, minfun, maxfun, meandom, mindom,maxdom, dfrange, modindex, bit_rate.
Lo script di estrazione di features, che è stato chiamato features_extraction.py, si occupa di estrarre le features per ogni file e salvare i risultati in 4 file csv separati.
In quanto i file sono approssimativamente 600.000 si è deciso di parallelizzare il processo in modo tale che i 4 file csv vengano scritti contemporaneamente. I risultati della parallelizzazione sono stati ben visibili: circa 4 giorni di lavoro senza parallelizzazione contro circa 2 giorni di lavoro con parallelizzazione.

### 
