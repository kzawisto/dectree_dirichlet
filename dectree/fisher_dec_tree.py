import sys, os
from dectree.learning import *
from dectree.base_tree import *
from dectree.algo_util import PriorityQueue
import numpy as np

import functools
import mc


@functools.total_ordering
class SplitLocation(object):
    def __init__(self, split_point, side, split_root, glob_root):
        assert side in ["left", "right"]
        self.split = split_point
        self.side = side
        self.root = split_root
        self.glob_root = glob_root

    def __lt__(self, other):
        return self.split.pvalue() < other.split.pvalue()

    def __eq__(self, other):
        return self.split.pvalue() == other.split.pvalue()

    def apply(self, features, target, sample_weight=None):
        if self.side == "left":
            selector = get_selector_for_node(self.glob_root, self.root.left_child, features)
            self.root.left_child = get_thump_out_of_split(self.split, features, target, selector, sample_weight)
            return self.root.left_child
        elif self.side == "right":
            selector = get_selector_for_node(self.glob_root, self.root.right_child, features)
            self.root.right_child = get_thump_out_of_split(self.split, features, target, selector, sample_weight)
            return self.root.right_child


def do_plotting(tree):
    import dectree.tree_draw as TD
    import matplotlib.pyplot as plt
    height = TD.set_h(tree)
    TD.set_offset(tree, height)

    fig, ax = plt.subplots(1, 1, figsize=(16, 14))

    TD.do_plotting(tree, ax)
    ax.grid()
    ax.autoscale()
    plt.show()


class FisherDecisionTree(object):
    def __init__(self, pvalue=0.9, max_depth=50, verbosity=0, kind="fisher"):
        self.verbosity = verbosity
        self.pvalue_limit = pvalue
        self.tree_initial = None
        self.pvalue = None
        self.max_depth = max_depth
        self.target_dim = None
        self.target_dtype = None
        self.kind = kind
        assert kind in ["fisher", "kolmogorov", "kolmogorov_split"], f"kind should be either fisher or kolmogorov, not {kind}"

    def get_best_split(self, features, target, root, node, father, side):
        selector = get_selector_for_node(root, node, features)
        positive_target_ratio = np.sum(target[selector], axis=0) / np.sum(target[selector])

        if (positive_target_ratio > 1 - 1e-8).any() or len(selector) < 10:
            return mc.List([])
        else:
            split = find_best_split(features, target, selector, kind=self.kind)
            return mc.List([SplitLocation(split, side, father, root)])

    def get_splits_for_node(self, root, node, features, target):
        return (
                self.get_best_split(features, target, root, node.right_child, node, "right") +
                self.get_best_split(features, target, root, node.left_child, node, "left")
        )

    def fit(self, features, target, sample_weight=None):
        self.target_dim = target.shape[1]
        assert self.target_dim == 2, "only target dim 2 supported, one hot encoded"
        self.target_dtype = target.dtype
        split = find_best_split(features, target, np.arange(target.shape[0]), kind=self.kind)
        if self.verbosity > 0:
            print(split)
        self.tree_initial = get_basic_thump(split.feature_num, split.value, 1, target, features,
                                            np.arange(features.shape[0]))
        pq = PriorityQueue(self.get_splits_for_node(self.tree_initial, self.tree_initial, features, target))
        self.pvalue = 1.0 - split.pvalue()

        if self.verbosity > 0:
            print(self.pvalue)
        for i in range(self.max_depth):
            split = pq.pop()
            self.pvalue *= (1 - split.split.pvalue())

            if self.pvalue < self.pvalue_limit:
                if self.verbosity > 0:
                    print(f"Next pval would be {self.pvalue}, final complexity {i-1}")
                self.pvalue /= (1 - split.split.pvalue())
                break

            if self.verbosity > 0:
                print(str(split.split), end="    ")
                print(self.pvalue, end="    ")
                print("que", mc.List(pq.container[0:3]).map(lambda x: x.split).map(str).mk_string())
            new_node = split.apply(features, target, sample_weight)
            pq.push_list(self.get_splits_for_node(self.tree_initial, new_node, features, target))

    def predict_proba(self, features):
        result = np.zeros((features.shape[0], self.target_dim), self.target_dtype)
        selector = np.arange(features.shape[0])
        self.tree_initial.predict_proba(features, result, selector)
        return result
