
import sys,os
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/..")
from dectree.learning import *
from dectree.base_tree import *
import numpy as np
np.random.seed(10)
features = np.random.randn(100,10)
target = np.eye(2)[np.random.randint(0, 2,100)]
split = find_best_split(features, target, np.arange(target.shape[0]))
tree_initial = get_basic_thump(split.feature_num, split.value, 1, target, features, np.arange(features.shape[0]))


print(split.impurity)
print(split.feature_num, split.value)
s = get_selector_for_node(tree_initial, tree_initial.right_child, features)
split2 = find_best_split(features, target, s)
print(split2.impurity, len(s))
s = get_selector_for_node(tree_initial, tree_initial.left_child, features)
split2 = find_best_split(features, target, s)
print(split2.impurity, len(s))
