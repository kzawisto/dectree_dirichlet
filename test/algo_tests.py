from nose.tools import eq_
from dectree.algo_util import PriorityQueue


def test_priority_queue():
    pq = PriorityQueue()
    pq.push(10)
    pq.push(12)
    pq.push(8)
    pq.push(6)
    pq.push(20)
    eq_(pq.pop(), 6)
    eq_(pq.pop(), 8)
    eq_(pq.pop(), 10)
    eq_(pq.pop(), 12)
    eq_(pq.pop(), 20)


def test_priority_queue_maintains_ascending_ord():
    pq = PriorityQueue()
    pq.push(10)
    pq.push(12)
    pq.push(8)
    pq.push(6)
    pq.push(20)
    eq_([6, 8, 10, 12, 20], pq.container)
