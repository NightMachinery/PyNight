import numpy

np = numpy
##
def uniform_from_rect(
    rng: numpy.random.Generator, shape, bounds_lower, bounds_upper
):
    """
    Returns an array with the given shape with elements sampled uniformly from within the given bounds.
    """

    bounds_length = bounds_upper - bounds_lower

    return bounds_lower + (rng.random(shape) * bounds_length)


##
