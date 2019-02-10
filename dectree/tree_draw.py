import matplotlib.pyplot as plt
import matplotlib.patches as P


class Graft(object):
    def __init__(self, v, lc=None, rc=None):
        self.v = v
        self.attr = {}
        self.left_child = lc
        self.right_child = rc

    def attach_left(self, lc):
        self.left_child = lc
        return self

    def attach_right(self, rc):
        self.right_child = rc
        return self

    def __str__(self):
        return str(self.v)


g = (
    Graft(10).attach_left(Graft(12).attach_left(Graft(3).attach_left(Graft(1))).attach_right(Graft(30))
                          ).attach_right(Graft(15).attach_left(Graft(8)).attach_right(Graft(6).attach_left(Graft(2))))
)


def set_h(g):
    g.attr["h"] = 1
    arr = [g]
    h_max = -1
    while len(arr) != 0:
        g = arr[0];
        arr = arr[1:]
        h_max = max(h_max, g.attr["h"])
        if g.left_child:
            g.left_child.attr["h"] = g.attr["h"] + 1
            arr.append(g.left_child)
        if g.right_child:
            g.right_child.attr["h"] = g.attr["h"] + 1
            arr.append(g.right_child)
    return h_max


def set_offset(g, height):
    initial_offset = 2 ** height
    g.attr["offset"] = initial_offset
    arr = [g]
    while len(arr) != 0:
        g = arr[0]
        arr = arr[1:]
        if g.left_child:
            g.left_child.attr["offset"] = g.attr["offset"] - 2 ** (height - g.attr["h"])
            arr.append(g.left_child)
        if g.right_child:
            g.right_child.attr["offset"] = g.attr["offset"] + 2 ** (height - g.attr["h"])
            arr.append(g.right_child)


def get_xy(g):
    return g.attr["h"] * 3, g.attr["offset"]


def do_plotting(g, ax):
    arr = [g]
    while len(arr) != 0:
        g = arr[0];
        arr = arr[1:]
        ax.add_patch(P.Circle(get_xy(g), 1))
        x, y = get_xy(g)
        ax.text(x, y, str(g), color="r")
        if g.left_child:
            x1, y1 = get_xy(g.left_child)
            ax.plot([x, x1], [y, y1], c="r")
            arr.append(g.left_child)
        if g.right_child:
            x1, y1 = get_xy(g.right_child)
            ax.plot([x, x1], [y, y1], c="r")
            arr.append(g.right_child)

