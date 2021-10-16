from itertools import tee

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..., (sn-2, sn-1), (sn-1, sn)"
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)