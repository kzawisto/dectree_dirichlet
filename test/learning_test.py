from dectree.learning import *
from dectree.maths import get_split_significance


def test_simple_split_case():
    feature = np.zeros(10)
    feature[::2] = [0.1, 0.2, 0.3, 0.4, 0.5]
    targets = np.eye(2)[np.arange(10) % 2]
    selector = np.arange(10)
    split_point = get_split_point(feature, targets, selector)
    assert split_point.bucket_size == 5


def test_pvalues():
    assert abs(get_split_significance(8, 1000, 5, 500) - 0.363) < 0.01



