import numpy

np = numpy
##
def uniform_from_rect(
    rng: numpy.random.Generator, shape, bounds_lower, bounds_upper
):
    """
    Returns an array with the given shape with elements sampled uniformly from within the given bounds.

    It can auto-infer the last dimension of 'shape' from the given bounds. (Use -1 to enable this.)

    Examples:
    >>> from numpy.random import default_rng
    >>> uniform_from_rect(default_rng(), (7,-1), np.array([-10, 0, 2]), np.array([-1, 4, 2]))
    array([[-6.85574093,  2.35340105,  2.        ],
       [-6.66585213,  2.55607637,  2.        ],
       [-1.3709884 ,  1.2369183 ,  2.        ],
       [-9.0566826 ,  2.3758122 ,  2.        ],
       [-4.01023858,  1.46456995,  2.        ],
       [-6.97080118,  3.22845693,  2.        ],
       [-7.10146632,  3.09328233,  2.        ]])
    """

    if shape[-1] == -1:
        shape = list(shape)  #: tuples are immutable
        shape[-1] = bounds_lower.shape[-1]
        shape = tuple(shape)

    bounds_length = bounds_upper - bounds_lower

    return bounds_lower + (rng.random(shape) * bounds_length)


##
