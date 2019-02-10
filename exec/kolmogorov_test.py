import numpy as np

import sys, os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")


def get_kstat(i, j):
    i = np.sort(i)
    j = np.sort(j)
    k_stat = np.searchsorted(i, j) / float(i.shape[0]) - np.arange(j.shape[0]) / float(j.shape[0])

    k_stat2 = k_stat * np.sqrt(i.shape[0] * j.shape[0] / (i.shape[0] + j.shape[0]))

    print(max(k_stat))
    return k_stat2


def get_kstat_w(i, iweight, j, jweight):
    ikey = np.argsort(i)
    jkey = np.argsort(j)
    iwcum = np.cumsum(iweight[ikey])
    iwcum -= iwcum[0]
    jwcum = np.cumsum(jweight[jkey])
    jwcum -= jwcum[0]
    jnormed = (jwcum / jwcum[-1])
    try:
        cum_proj = iwcum[np.clip(np.searchsorted(i[ikey], j[jkey]), 0, len(iwcum) - 1)] / iwcum[-1]
        k_stat = cum_proj - jnormed
        print(max(k_stat))
    except IndexError as e:
        print(e)
        import pdb;
        pdb.set_trace()
    return np.abs(k_stat) * np.sqrt(i.shape[0] * j.shape[0] / (i.shape[0] + j.shape[0]))


def get_bootstrap(N, K=1000):
    result = []
    for k in range(N):
        i = np.random.randn(1000)
        j = np.random.randn(1000)
        result.append(np.max(get_kstat(i, j)))
    return sorted(result)


# bs = get_bootstrap(100,1000)

from scipy.stats import ks_2samp

actual = []
expected = []
for l in range(1):
    i = np.random.randn(10000)
    j = np.random.randn(10000)
    print(j[0:10])
    kstat = max(get_kstat(i, j)) / 2

    print(j[0:10])
    ks2 = ks_2samp(i, j)[0]

    print(j[0:10])
    ks3 = max(get_kstat_w(i, np.ones(len(i)) * 1.0, j, np.ones(len(j)) * 1.0)) / 2
    print(kstat, ks2, ks3)
    print(ks_2samp(i, j)[1])
    actual.append(kstat)
    expected.append(ks2)


class KolmogorovTestStats(object):
    def __init__(self, target, chosen_feature):
        one, other = chosen_feature[target[:, 0] == 1], chosen_feature[target[:, 0] != 1]

        if one.shape[0] != 0 and other.shape[0] != 0:
            self.pvalue = ks_2samp(one, other)[1]
        else:
            self.pvalue = 1

    def get_pvalue(self):
        return self.pvalue

    def __str__(self):
        return f'KS({self.get_pvalue()})'

from collections import namedtuple

KsResult = namedtuple("KsResult",("proba", "d","idx"))


target_ = np.eye(2)[np.random.randint(0, 1, 200)]
#target_ = [0,1]
print(target_)
print(
    KolmogorovTestStats(
        target_,
        np.random.rand(200)
    ))

# import matplotlib.pyplot as plt

# plt.scatter(actual,expected)
# plt.show()
