import numpy as np


def get_symmetric_pvalue(value):
    return 0.5 - np.abs(value - 0.5)


def get_split_significance(split_point, sample_size, hit_positives, all_positives):
    import scipy.stats as ss
    return get_symmetric_pvalue(
        ss.hypergeom.cdf([hit_positives - 1, hit_positives + 1], sample_size, all_positives, split_point)
    ).max()