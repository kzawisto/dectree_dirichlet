import numpy as np
import pandas as pd


class TreeComponent(object):
    def __init__(self):
        pass

    def predict_proba(self, features, result, selector):
        pass


from scipy.stats import ks_2samp
class TreeLeaf(TreeComponent):
    def __init__(self, proba):
        super().__init__()
        self.attr = {}
        self.proba = proba
        self.left_child = None
        self.right_child = None

    def predict_proba(self, features, result, selector):
        self.attr["last_num"] = len(selector)
        result[selector] = self.proba

    def __str__(self):
        count = 0
        if "last_num" in self.attr:
            count = self.attr["last_num"]
        return "Leaf(" + np.array2string(self.proba, precision=3) + "," +str(count) + ")"


class TreeGraft(TreeComponent):
    def __init__(self, feature_num, threshold, label, left_child, right_child):
        super().__init__()
        self.attr = {}
        self.feature_num = feature_num
        self.threshold = threshold
        self.label = label
        self.left_child = left_child
        self.right_child = right_child

    def apply_selector(self, features, selector, left=True):
        next_selector = (features[selector, self.feature_num] >= self.threshold)
        if left:
            return selector[~next_selector]
        else:
            return selector[next_selector]

    def predict_proba(self, features, result, selector):

        self.attr["last_num"] = len(selector)
        next_selector = (features[selector, self.feature_num] >= self.threshold)
        self.left_child.predict_proba(features, result, selector[~next_selector])
        self.right_child.predict_proba(features, result, selector[next_selector])

    def __str__(self):
        count = 0
        if "last_num" in self.attr:
            count = self.attr["last_num"]
        return f"Graft(thr={self.threshold:.4}, feature={self.feature_num}, c ={count})"


def get_probas(target):

    return target.sum(axis=0) / target.sum()

def get_probas_weighted(target, weight):
    t = target * weight
    return t.sum(axis=0) / t.sum()



class DecTree(object):
    def __init__(self, tree_elem, target_dim):
        self.tree_elem: TreeComponent = tree_elem
        self.target_dim = target_dim

    def predict(self, features):
        result = np.zeros((features.shape[0], self.target_dim), dtype=features.dtype)
        selector = np.arange(0, features.shape[0])
        self.tree_elem.predict_proba(features, result, selector)
        return result


def find_path_to_elem(root, element_to_find):
    stack = [root]
    parent_map = {}

    while True:
        elem = stack[-1]
        stack.pop()
        if elem == element_to_find:
            break
        if elem.right_child:
            stack.append(elem.right_child)
            parent_map[elem.right_child] = elem
        if elem.left_child:
            stack.append(elem.left_child)
            parent_map[elem.left_child] = elem

    elem_iter = element_to_find
    result = [elem_iter]
    while elem_iter in parent_map:
        elem_iter = parent_map[elem_iter]
        result.append(elem_iter)
    return result


def get_selector_for_node(root, node, features):
    path = find_path_to_elem(root, node)[::-1]
    selector = np.arange(features.shape[0])
    for (this, nex) in zip(path[:-1], path[1:]):
        is_left = this.left_child == nex
        if not is_left:
            assert this.right_child == nex
        selector = this.apply_selector(features, selector, is_left)
    return selector
