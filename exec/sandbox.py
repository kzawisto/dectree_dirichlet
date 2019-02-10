import sys, os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from dectree.fisher_dec_tree import FisherDecisionTree
import numpy as np



np.random.seed(10)
features_ = np.random.randn(40000, 40)
target_ = np.eye(2)[np.random.randint(0, 2, 40000)]

features_[:, 7] += target_[:, 0]
features_[:, 3] += target_[:, 0]
features_[:, 5] += target_[:, 0]
features_[:, 2] += target_[:, 0]
features_[:, 1] -= target_[:, 0]
tree = FisherDecisionTree(0.9, 300,  verbosity=1, kind="kolmogorov")
print("starting fit")
tree.fit(features_, target_)

result = tree.predict_proba(features_)

print((result.argmax(axis=1) == target_.argmax(axis=1)).sum() / len(target_))

"""
print(split.feature_num, split.value)
s = get_selector_for_node(tree_initial, tree_initial.right_child, features)
split2 = find_best_split(features, target, s)
print(split2.impurity, len(s), split2.ft_stats.get_pvalue())

s = get_selector_for_node(tree_initial, tree_initial.left_child, features)
split2 = find_best_split(features, target, s)
print(split2.impurity, len(s), split2.ft_stats.get_pvalue())
"""
