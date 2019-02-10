
import sys, os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from dectree.fisher_dec_tree import FisherDecisionTree
import numpy as np



np.random.seed(10)
features_ = np.random.randn(4000, 40)
target_ = np.eye(2)[np.random.randint(0, 2, 4000)]

features_[:, 1] -= target_[:, 0]
tree = FisherDecisionTree(0.9, 30,  verbosity=0)
print("starting fit")
tree.fit(features_, target_)

result = tree.predict_proba(features_)

print((result.argmax(axis=1) == target_.argmax(axis=1)).sum() / len(target_))

