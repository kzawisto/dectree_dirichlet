from dectree.base_tree import *


def test_it_should_do_prediction_for_one_graft():
    l1 = TreeLeaf(np.array([0.1, 0.9]))
    l2 = TreeLeaf(np.array([0.8, 0.2]))
    graft = TreeGraft(1, 0.5, "wuj", l1, l2)
    dectr = DecTree(graft, target_dim=2)
    features = np.zeros((20, 3))
    features[:5, 1] = 2
    result = dectr.predict(features)
    assert (result[:5] == [0.8, 0.2]).all()
    assert (result[5:] == [0.1, 0.9]).all()


def test_it_should_do_prediction_for_two_grafts():
    l1 = TreeLeaf(np.array([0.1, 0.9]))
    l2 = TreeLeaf(np.array([0.8, 0.2]))
    l3 = TreeLeaf(np.array([0.5, 0.5]))
    graft = TreeGraft(1, 0.5, "wuj", l1, l2)
    graft2 = TreeGraft(2, 0.3, "wuj", l3, graft)
    dectr = DecTree(graft2, target_dim=2)
    features = np.zeros((20, 3))
    features[:10, 2] = 0.5
    features[5:, 1] = 0.5

    result = dectr.predict(features)
    assert (result[:5] == [0.1, 0.9]).all()
    assert (result[5:10] == [0.8, 0.2]).all()
    assert (result[10:] == [0.5, 0.5]).all()


def test_it_should_find_selector():
    l1 = TreeLeaf(np.array([0.1, 0.9]))
    l2 = TreeLeaf(np.array([0.8, 0.2]))
    l3 = TreeLeaf(np.array([0.5, 0.5]))
    graft = TreeGraft(1, 0.5, "wuj", l1, l2)
    graft2 = TreeGraft(2, 0.3, "wuj", l3, graft)
    features = np.zeros((20, 3))
    features[:10, 2] = 0.5
    features[5::2, 1,] = 0.5
    assert ([5, 7, 9] == get_selector_for_node(graft2, l2, features)).all()
