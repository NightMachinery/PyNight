import numpy
np = numpy
##
def partition_int_into_fixed(n, boxes):
    "Partitions the integer 'n' into 'boxes' boxes, each box having at least one element. Order matters. Returns a 2D numpy array."

    n -= boxes #: give each box one
    partition_0 = list(partition_int_into_fixed_0(n, boxes))
    partition_0 = np.array(partition_0)
    partition = partition_0 + 1
    return partition

def partition_int_into_fixed_0(n, boxes):
    "Generates the partitions of the integer 'n' into 'boxes' boxes. Boxes can have zero elements. Order matters."

    if boxes == 0:
        yield []
    elif boxes == 1:
        yield [n]
    else:
        for first in range(n+1):
            partitions = partition_int_into_fixed_0(n - first, boxes - 1)
            for partition in partitions:
                yield [first] + partition
##
