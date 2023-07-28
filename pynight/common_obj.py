##
def delattr_force(obj, attr):
    """
    Delete an attribute from an object if it exists.

    Parameters:
    obj: The object from which the attribute should be deleted.
    attr: The name of the attribute to delete.
    """
    if hasattr(obj, attr):
        delattr(obj, attr)


##
