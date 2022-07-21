from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis 

FEATURES = ["bfcc",
            "lfcc",
            "lpc",
            "lpcc",
            "mfcc",
            "imfcc",
            "msrcc",
            "ngcc",
            "psrcc",
            "plp",
            "rplp",
            "mel_filter_banks",
            "bark_filter_banks",
            "gammatone_filter_banks",
            "spectrum",
            "mean_frequency",
            "peak_frequency",
            "frequencies_std",
            "amplitudes_cum_sum",
            "mode_frequency",
            "median_frequency",
            "frequencies_q25",
            "frequencies_q75",
            "iqr",
            "freqs_skewness",
            "freqs_kurtosis",
            "spectral_entropy",
            "spectral_flatness",
            "spectral_centroid",
            "spectral_bandwidth",
            "spectral_spread",
            "spectral_rolloff",
            "energy",
            "rms",
            "zcr",
            "spectral_mean",
            "spectral_rms",
            "spectral_std",
            "spectral_variance",
            "meanfun",
            "minfun",
            "maxfun",
            "meandom",
            "mindom",
            "maxdom",
            "dfrange",
            "modindex",
            "bit_rate"]

SAMPLE_ROWS_STATS = 10000

ROWS_CREATION_MODELS = 2000

CLASSIFIERS = [ DecisionTreeClassifier,
        SVC,
        LogisticRegression,
        KNeighborsClassifier,
        LinearDiscriminantAnalysis,
        GaussianProcessClassifier,
        RandomForestClassifier,
        MLPClassifier,
        AdaBoostClassifier,
        GaussianNB,
        MultinomialNB,
        QuadraticDiscriminantAnalysis
]

VARIABLES_TO_DROP = [
        'amplitudes_cum_sum',
        'bark_filter_banks',
        'dfrange',
        'energy',
        'freqs_kurtosis',
        'freqs_skewness',
        'frequencies_q25',
        'frequencies_q75',
        'frequencies_std',
        'gammatone_filter_banks',
        'iqr',
        'maxfun',
        'mean_frequency',
        'meandom',
        'meanfun',
        'median_frequency',
        'mel_filter_banks',
        'mindom',
        'minfun',
        'mode_frequency',
        'modindex',
        'msrcc',
        'peak_frequency',
        'rms',
        'spectral_bandwidth',
        'spectral_centroid',
        'spectral_entropy',
        'spectral_flatness',
        'spectral_mean',
        'spectral_rms',
        'spectral_spread',
        'spectral_std',
        'spectrum',
        'spectral_variance',
        'zcr']
