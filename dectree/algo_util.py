
import heapq


class PriorityQueue(object):
    def __init__(self, arg=None):
        if arg is None:
            arg = []
        self.container = arg

    def push(self, elem):
        return heapq.heappush(self.container, elem)

    def push_list(self, elems):
        for e in elems:
            self.push(e)

    def pop(self):
        return heapq.heappop(self.container)

    def empty(self):
        return len(self.container) == 0

