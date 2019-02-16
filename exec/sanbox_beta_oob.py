import sys, os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from dectree.beta_dec_tree import BetaDecisionTreeOOB
import numpy as np



np.random.seed(10)
features_ = np.random.randn(40000, 40)
target_ = np.eye(2)[np.random.randint(0, 2, 40000)]

features_[:, 7] += target_[:, 0]
features_[:, 3] += target_[:, 0]
features_[:, 5] += target_[:, 0]
features_[:, 2] += target_[:, 0]
features_[:, 1] -= target_[:, 0]
tree = BetaDecisionTreeOOB({"pval":1e-7,"depth":30, "oob_ratio":0.2})
import scipy.stats

scipy.stats.beta
print("starting fit")
tree.fit(features_, target_)

result = tree.predict_proba(features_)

print((result.argmax(axis=1) == target_.argmax(axis=1)).sum() / len(target_))
