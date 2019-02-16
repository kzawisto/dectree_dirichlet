
import numpy as np
import scipy.stats as SST
import scipy.special as SSP,numpy as np, pandas as pd, matplotlib.pyplot as plt

from dectree.base_tree import get_selector_for_node
from dectree.learning import get_thump_out_of_split

np.set_printoptions(suppress=True)
from dectree.algo_util import PriorityQueue
import numpy as np

import functools
import mc



def bivar_argmax(arr):
    assert len(arr.shape)==2
    return np.unravel_index(np.argmax(arr),arr.shape)

@functools.total_ordering
class BetaSplit(object):
    def __init__(self,position, score, cutoff,prior =None):
        self.score = score
        self.position = position
        self.value = cutoff# cutoff
        self.prior = prior
        self.feature_num = -1

    def with_feature_num(self,i):
        self.feature_num = i
        return self

    def __str__(self):
        return f"{self.score}_{self.position}_{self.feature_num}_{self.value}"

    def __lt__(self, other):
        return self.score > other.score

    def __eq__(self, other):
        return self.score == other.score


def get_split_vals_approx(x,y,pval)->BetaSplit:
    idx = np.argsort(x)
    if len(y.shape)==2:
        y = y[:,0]
    stride = 1
    if len(y) > 10000 and len(y)<100000:
        stride=50
    if len(y) > 100000:
        stride=500


    a = y[idx]

    k = np.cumsum(a)
    n_k = np.concatenate([np.array([0]), np.cumsum(y[idx[1:]][::-1])])[::-1]  # + np.cumsum(a)
    # /(np.arange(len(a))+1)
    num_st = (np.arange(len(a)) + 1)
    num_2nd = (np.arange(len(a)))[::-1]
    num_2nd[-1] = 1
    if stride !=1:

        k =k[::stride]
        num_st =num_st[::stride]
        n_k = n_k[::stride]
        num_2nd = num_2nd[::stride]
        # /(np.arange(len(a))+1)

    prior=(np.mean(y))

    prior_upper= SST.beta(k[-1]+1, num_st[-1]+1).ppf(1-pval)
    prior_lower=SST.beta(k[-1]+1, num_st[-1]+1).ppf(pval)
    results= np.stack([
        SST.beta(k + 1, num_st - k + 1).ppf(pval)-prior_upper,
        # prior_lower-SST.beta(k + 1, num_st - k + 1).ppf(1 - pval),
        SST.beta(n_k + 1, num_2nd - n_k + 1).ppf(pval)-prior_upper,
        # prior_lower-SST.beta(n_k + 1, num_2nd - n_k + 1).ppf(1-pval)
    ])
    # bivar_argmax(arr)
    arg=bivar_argmax(results)
    vals=results[arg]

    arg = (arg[0], arg[1]*stride)
    if arg[1] +1 < len(x):
        cutoff = x[idx][arg[1]:arg[1]+2].mean()+1e-9
    else:
        cutoff = x[idx][arg[1]]+1e-9
    # print(cutoff)
    # print(a[arg[1]:arg[1]+2])
    return BetaSplit(arg, vals,cutoff, prior_upper)


def get_split_approx(k,num_st, n_k, num_2nd,prior,pval):
    m1 = get_approx_max(k+1, num_st - k + 1,10000,lambda x,y: SST.beta(x,y).ppf(pval)-prior) +(0)
    m2 = get_approx_max(k + 1, num_st - k + 1,10000,lambda x,y: prior - SST.beta(x,y).ppf(1-pval))+(1)
    m3 = get_approx_max(n_k+1, num_2nd - k + 1,10000,lambda x,y: SST.beta(x,y).ppf(pval)-prior)+(2)
    m4 = get_approx_max(k+1, num_2nd - k + 1,10000,lambda x,y: prior - SST.beta(x,y).ppf(1-pval))+(3)
    return mc.List([m1,m2,m3,m4]).map(lambda x: (x[0],(x[1],x[2]))).sort(lambda x: x[0])[-1]

def get_split_vals(x,y,pval)->BetaSplit:
    idx = np.argsort(x)
    if len(y.shape)==2:
        y = y[:,0]
    a = y[idx]

    k = np.cumsum(a)
    n_k = np.concatenate([np.array([0]), np.cumsum(y[idx[1:]][::-1])])[::-1]  # + np.cumsum(a)
    # /(np.arange(len(a))+1)
    num_st = (np.arange(len(a)) + 1)
    num_2nd = (np.arange(len(a)))[::-1]
    num_2nd[-1] = 1

    prior_upper= SST.beta(k[-1]+1, num_st[-1]+1).ppf(1-pval)
    prior_lower=SST.beta(k[-1]+1, num_st[-1]+1).ppf(pval)
    results= np.stack([
        SST.beta(k + 1, num_st - k + 1).ppf(pval)-prior_upper,
        prior_lower-SST.beta(k + 1, num_st - k + 1).ppf(1 - pval),
        SST.beta(n_k + 1, num_2nd - n_k + 1).ppf(pval)-prior_upper,
        prior_lower-SST.beta(n_k + 1, num_2nd - n_k + 1).ppf(1-pval)
    ])
    # bivar_argmax(arr)
    arg=bivar_argmax(results)
    vals=results[arg]
    if arg[1] +1 < len(x):
        cutoff = x[idx][arg[1]:arg[1]+2].mean()+1e-9
    else:
        cutoff = x[idx][arg[1]]+1e-9
    # print(cutoff)
    # print(a[arg[1]:arg[1]+2])
    return BetaSplit(arg, vals,cutoff)

def get_approx_max(one, other,number_of_steps,bivar_ufunc)->tuple:
    offset=0
    upper = one.shape[0]
    maxval = np.nan
    for i in range(100):
        stride= int(np.ceil((upper -offset)/number_of_steps))

        initial_result = bivar_ufunc(one[offset:upper:stride],other[offset:upper:stride])
        max1= np.argmax(initial_result)
        maxval = initial_result[max1]
        max2= offset+stride*max1
        offset,upper=np.max([max2 - stride,0]),np.min([max2 + stride,upper])
        if stride==1:
            break
    return maxval, max2



@functools.total_ordering
class BetaSplitLocation(object):
    def __init__(self, split_point, side, split_root, glob_root):
        assert side in ["left", "right"]
        self.split : BetaSplit = split_point
        self.side = side
        self.root = split_root
        self.glob_root = glob_root

    def __lt__(self, other):
        return self.split.score+self.split.prior > other.split.score + other.split.prior

    def __eq__(self, other):
        return self.split.score+other.split.prior == other.split.score+other.split.prior

    def emptiness_check(self,features):
        if self.side == "left":
            selector1 = get_selector_for_node(self.glob_root, self.root.left_child, features)
            selector2 = get_selector_for_node(self.glob_root, self.root, features)
            print(f"splitting {selector2.shape[0]} into {selector1.shape[0]} and {selector2.shape[0] - selector1.shape[0]}")
            if selector1.shape == selector2.shape or selector1.shape[0]==0:
                print("empty split - skipping")
                return False
            else:
                return True

        elif self.side == "right":
            selector1 = get_selector_for_node(self.glob_root, self.root.right_child, features)
            selector2 = get_selector_for_node(self.glob_root, self.root, features)

            print(f"splitting {selector2.shape[0]} into {selector1.shape[0]} and {selector2.shape[0] - selector1.shape[0]}")

            if selector1.shape == selector2.shape or selector1.shape[0]==0:
                print("empty split - skipping")
                return False
            else:
                return True
        else:
            raise Exception("illegal arg side "+ str(self.side))
    def apply(self, features, target, sample_weight=None):
        if self.side == "left":
            selector = get_selector_for_node(self.glob_root, self.root.left_child, features)
            self.root.left_child = get_thump_out_of_split(self.split, features, target, selector, sample_weight)
            return self.root.left_child
        elif self.side == "right":
            selector = get_selector_for_node(self.glob_root, self.root.right_child, features)
            self.root.right_child = get_thump_out_of_split(self.split, features, target, selector, sample_weight)
            return self.root.right_child
    # def apply(self, features, target):
    #     if self.side == "left":
    #         selector = get_selector_for_node(self.glob_root, self.root.left_child, features)
    #         self.root.left_child = get_thump_out_of_split(self.split, features, target, selector, sample_weight)
    #         return self.root.left_child
    #     elif self.side == "right":
    #         selector = get_selector_for_node(self.glob_root, self.root.right_child, features)
    #         self.root.right_child = get_thump_out_of_split(self.split, features, target, selector, sample_weight)
    #         return self.root.right_child

if __name__ == "__main__":

    np.random.seed(1)
    pval = 1e-3
    X=np.random.randn(1000,10)
    y = np.zeros(1000)
    # y=np.random.randint(0,2,    1000)
    y[X[:,0]>0.89]=1
    y[X[:,2]<-1.53]=1

    import scipy.stats, numpy as np


    x = X[:, 0]
    print(
        get_split_vals(x,y,1e-3)
    )
    x = X[:, 2]
    print(
        get_split_vals(x,y,1e-3)
    )
