import numpy as np, mc

from dectree.algo_util import PriorityQueue
from dectree.base_tree import get_selector_for_node
from dectree.betatree import BetaSplitLocation,get_split_vals, get_split_vals_approx, BetaSplit
from dectree.learning import find_best_split, get_basic_thump


def get_split_point(x,y, selector,cfg):
    return get_split_vals_approx(x[selector],y[selector],cfg["pval"])

def find_best_split_beta(features, target, selector,cfg,verbose=0)->BetaSplit:
    split_points = [get_split_point(features[:, i], target, selector,cfg).with_feature_num(i) for i in
                    range(features.shape[1])]

    split_points = sorted(split_points, key=lambda s: s.score)

    if verbose > 0:
        print("candidates")
        print(split_points[-6:])
    return split_points[-1]

class BetaDecisionTree(object):
    def __init__(self, cfg):
        self.tree_initial = None
        self.pvalue = None
        self.cfg = cfg
        self.target_dim = None
        self.target_dtype = None
        self.verbosity = 1
        self.max_depth = cfg["depth"]

    def get_best_split(self, features, target, root, node, father, side):
        selector = get_selector_for_node(root, node, features)

        if len(selector) < 10:
            return mc.List([])
        else:
            split = find_best_split_beta(features, target, selector,self.cfg)
            if split.score > 0:
                return mc.List([BetaSplitLocation(split, side, father, root)])
            else:
                return mc.List([])

    def get_splits_for_node(self, root, node, features, target):
        return (
            self.get_best_split(features, target, root, node.right_child, node, "right") +
            self.get_best_split(features, target, root, node.left_child, node, "left")
        )

    def fit(self, features, target, sample_weight=None):
        self.target_dim = target.shape[1]
        assert self.target_dim == 2, "only target dim 2 supported, one hot encoded"
        self.target_dtype = target.dtype
        split = find_best_split_beta(features, target, np.arange(target.shape[0]),self.cfg)
        if self.verbosity > 0:
            print(split)
        self.tree_initial = get_basic_thump(split.feature_num, split.value, 1, target, features,
                                            np.arange(features.shape[0]))
        pq = PriorityQueue(self.get_splits_for_node(self.tree_initial, self.tree_initial, features, target))

        if self.verbosity > 0:
            print(self.pvalue)
        for i in range(self.max_depth):
            if pq.empty():
                break
            split = pq.pop()

            if self.verbosity > 0:
                print("\nn split is",str(split.split), end="\n")
                print("\nqueue is ", mc.List(pq.container[0:3]).map(lambda x: x.split).map(str).mk_string())
            new_node = split.apply(features, target, sample_weight)
            pq.push_list(self.get_splits_for_node(self.tree_initial, new_node, features, target))

    def predict_proba(self, features):
        result = np.zeros((features.shape[0], self.target_dim), self.target_dtype)
        selector = np.arange(features.shape[0])
        self.tree_initial.predict_proba(features, result, selector)
        return result
