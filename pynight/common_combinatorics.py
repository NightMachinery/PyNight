import numpy

np = numpy
##
def partition_int_into_fixed_gen(n, boxes, min=1):
    "Partitions the integer 'n' into 'boxes' boxes, each box having at least 'min' elements. Order matters. Returns a generator."
    if boxes == 0:
        if n == 0:
            yield tuple()

        return
    elif boxes < 0:
        return
    elif boxes == 1:
        if n >= min:
            yield (n,)

        return
    else:
        for i in range(min, n + 1):
            for result in partition_int_into_fixed_gen(
                n - i, boxes - 1, min
            ):
                yield (i,) + result


def partition_int_into_fixed(n, boxes, min=1):
    "Partitions the integer 'n' into 'boxes' boxes, each box having at least 'min' elements. Order matters. Returns a 2D numpy array."

    partitions = list(partition_int_into_fixed_gen(n, boxes, min=min))
    partitions = np.array(partitions, dtype=np.int64)
    return partitions


##
def permutations_gen(lst):
    #: @todo optionally, remove duplicate permutations
    #:
    #: Tests:
    #: `len(list(permutations_gen(range(5)))) == 120`
    ##
    l = len(lst)
    if l <= 1:
        yield lst
        return
    else:
        for i in range(l):
            last = lst[i]
            for p in permutations_gen(list(lst[:i]) + list(lst[i + 1 :])):
                p.append(last)
                yield p


##
