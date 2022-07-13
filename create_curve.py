import numpy as np
from opt_bins import opt_bins #method found by Knuth to calculate optimal number of bins
# opt_bins is slower respect to freedman_diaconis
from typing import Union

def freedman_diaconis(values: list) -> int:
    # it uses Freedman Diaconis rule
    # https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule
    q25, q75 = np.percentile(values, [25, 75])
    bin_width = (2 * (q75 - q25)) / (len(values)**(1 / 3))
    values = np.array(values)
    if bin_width == 0:
        bin_width = 50
    # Freedman Diaconis rule does not calculate number of bins, it calculates bin_width
    # We can get number of bins by bin_width using this formula
    bins = round((values.max() - values.min()) / bin_width)
    if bins == 0:
        return 50
    return bins

def generate_hist(data: list, method='freedman_diaconis') -> np.ndarray:
    bins = 50
    if method == 'freedman_diaconis':
        bins = freedman_diaconis(data)
    elif method == 'knuth':
        bins = opt_bins(data,len(data))
    
    hist, __ = np.histogram(data, bins=bins)
    return  hist

def create_curve(values: list, resize=None, method='freedman_diaconis') -> Union[np.ndarray, int]:
    for idx in range(len(values)):
        if np.isnan(values[idx]):
            values[idx] = 0
    if resize:
        values = values[:resize]
    hist = generate_hist(values,method)
    list_size = str(len(values))
    #if we plot hist using plt.plot(), it will be similar to a curve
    return hist, list_size