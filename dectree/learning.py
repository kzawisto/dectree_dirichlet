import numpy as np

from dectree.maths import get_split_significance
from dectree.base_tree import *


def gini_impurity(target):
    v = 1 - ((target / target.sum(axis=1).reshape(-1, 1)) ** 2).sum(axis=1)
    return v


def impurity_single(target):
    return 1 - ((target.sum(axis=1) / target.sum()) ** 2).sum(axis=1)


class FisherTestStats(object):
    def __init__(self, split_size, hit_positives, sample_size, sample_positives):
        self.split_size = split_size
        self.hit_positives = hit_positives
        self.sample_size = sample_size
        self.sample_positives = sample_positives

    def get_pvalue(self):
        return get_split_significance(split_point=self.split_size, sample_size=self.sample_size,
                                      hit_positives=self.hit_positives, all_positives=self.sample_positives)

    def __str__(self):
        return f'FTStat({self.split_size}, {self.sample_size}, {self.get_pvalue()})'



class KolmogorovTestStats(object):
    def __init__(self, target, chosen_feature):
        from scipy.stats import ks_2samp
        one, other = chosen_feature[target[:, 0] == 1], chosen_feature[target[:, 0] != 1]

        if one.shape[0] != 0 and other.shape[0] != 0:
            self.pvalue = ks_2samp(one, other)[1]
        else:
            self.pvalue = 1

    def get_pvalue(self):
        return self.pvalue

    def __str__(self):
        return f'KS({self.get_pvalue()})'

class SplitPoint(object):
    def __init__(self, value, bucket_size, impurity, ft_stats=None):
        self.value = value
        self.bucket_size = bucket_size
        self.impurity = impurity
        self.feature_num = None
        self.ft_stats = ft_stats

    def with_feature_num(self, i):
        self.feature_num = i
        return self

    def pvalue(self):
        return self.ft_stats.get_pvalue()

    def __str__(self):
        return self.ft_stats.__str__()


def get_split_point(feature, target, selector, kind="fisher"):

    chosen_feature = feature[selector]

    sorted_keys = np.argsort(chosen_feature)
    target_select = target[selector][sorted_keys]

    target_cum = target_select.cumsum(axis=0)
    target_cum_rev = target_select[::-1].cumsum(axis=0)[::-1]
    impurity = (target_cum_rev[1:].sum(axis=1) * gini_impurity(target_cum_rev[1:]) +
                target_cum[:-1].sum(axis=1) * gini_impurity(target_cum[:-1])) / target_cum.shape[0]
    split_point = np.clip(np.argmin(impurity) + 1, 0, len(selector) - 1)

    if kind == "fisher":
        ft_stats = FisherTestStats(split_point, target_cum[split_point, 0], len(selector), target_cum[-1, 0])
    elif kind == "kolmogorov":
        ft_stats = KolmogorovTestStats(target[selector], chosen_feature)
    else:
        raise Exception(f"kind={kind} instead of fisher and kolmogorov.")
    return SplitPoint(chosen_feature[sorted_keys][split_point], split_point, np.min(impurity), ft_stats)


def find_best_split(features, target, selector, kind="fisher"):
    split_points = [get_split_point(features[:, i], target, selector, kind).with_feature_num(i) for i in
                    range(features.shape[1])]
    split_points = sorted(split_points, key=lambda s: s.impurity)
    return split_points[0]


class ConstructionState(object):
    def __init__(self):
        self.root = None
        self.selector = None
        self.deep = 0


def get_basic_thump(feature_num, threshold, label, target, features, selector):
    next_selector = (features[selector, feature_num] >= threshold)
    proba1 = get_probas(target[selector[next_selector]])
    right = TreeLeaf(proba1)
    proba2 = get_probas(target[selector[~next_selector]])
    left = TreeLeaf(proba2)
    return TreeGraft(feature_num, threshold, label, left, right)


def get_weighted_thump(feature_num, threshold, label, target, features, selector, sample_weight):
    next_selector = (features[selector, feature_num] >= threshold)
    proba1 = get_probas_weighted(target[selector[next_selector]], sample_weight[selector[next_selector]])
    right = TreeLeaf(proba1)
    proba2 = get_probas_weighted(target[selector[~next_selector]], sample_weight[selector[~next_selector]])
    left = TreeLeaf(proba2)
    return TreeGraft(feature_num, threshold, label, left, right)


def get_thump_out_of_split(split, features, target, selector, sample_weight=None):
    if None is sample_weight:
        return get_basic_thump(split.feature_num, split.value, "", target, features, selector)
    else:
        return get_weighted_thump(split.feature_num, split.value, "", target, features, selector, sample_weight)


#   tree_initial =
# TreeGraft(split.feature_num, split.value, "", None, None)
