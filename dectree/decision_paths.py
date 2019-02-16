import dectree.base_tree as BT

import mc
import numpy as np


def drop_key(dic, key):
    if key in dic:
        del dic[key]


def drop_fr_dic(dic):
    import copy
    dic = copy.copy(dic)
    drop_key(dic, "father")
    drop_key(dic, "L")
    drop_key(dic, "R")
    return dic

def jsonify(graft, columns= None):
    dic={}
    if isinstance(graft, BT.TreeGraft):
        dic["cutoff"] = graft.threshold
        if columns is not None:
            dic["feature"] = columns[graft.feature_num]
        dic["fnum"] = graft.feature_num
        dic["L"] = jsonify(graft.left_child)
        dic["R"] = jsonify(graft.right_child)

    if isinstance(graft, BT.TreeLeaf):
        dic["proba"] = graft.proba
    return dic


def set_fathers(dic):
    if "proba" in dic:
        pass
    else:
        dic["L"]["father"] = dic
        dic["L"]["side"] = "left"
        set_fathers(dic["L"])
        dic["R"]["father"] = dic
        dic["R"]["side"] = "right"

        set_fathers(dic["R"])


def get_all_leaves(dic_graft, leaves=None):
    if leaves is None:
        leaves = []
    if "proba" in dic_graft:
        leaves.append(mc.Dict(dic_graft))
    else:
        get_all_leaves(dic_graft["L"], leaves)
        get_all_leaves(dic_graft["R"], leaves)

    return leaves


def node_to_path(node):
    import copy
    node = copy.copy(node)
    arr = mc.List([node])
    while "father" in node:
        fath = copy.copy(node["father"])
        if node["side"] == "right":
            fath["sign"] = ">"
        if node["side"] == "left":
            fath["sign"] = "<"
        node = fath

        arr.append(mc.Dict(node))
    return arr[::-1]


def path_redundancy_removal(nodes):
    dic = {}
    for n in nodes[:-1]:
        import copy
        n = copy.copy(n)
        drop_key(n, "side")
        key = n["feature"] + n["sign"]
        if key not in dic:
            dic[key] = []
        dic[key].append(n)
    for key in dic:
        if dic[key][0]['sign'] == "<":
            dic[key] = sorted(dic[key], key=lambda x: x["cutoff"])
        elif dic[key][0]['sign'] == ">":
            dic[key] = sorted(dic[key], key=lambda x: -x["cutoff"])
        else:
            raise Exception("")
    return [dic[key][0] for key in dic] + nodes[-1:]


def path_predict(path, features):
    selector = np.ones(features.shape[0], dtype="int")
    prediction = np.zeros(features.shape[0])
    for p in path[:-1]:
        if p["sign"] == "<":
            selector[~(p["cutoff"] > features[:, p["fnum"]])] = 0
        if p["sign"] == ">":
            selector[~(p["cutoff"] <= features[:, p["fnum"]])] = 0
    prediction[selector] = 1
    return selector

def get_path_model(model, columns=None):
    graft=model.tree_initial
    dic = jsonify(graft,columns)
    set_fathers(dic)
    leaves= mc.List(get_all_leaves(dic))
    leaves=leaves.filter(lambda x: not np.isnan(x["proba"][0])).sorted(key=lambda x: x["proba"][0])

    path=mc.List(node_to_path(leaves[-1])).map(drop_fr_dic)
    path_opt = path_redundancy_removal(path)
    return path_opt


