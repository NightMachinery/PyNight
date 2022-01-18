##
def iterable_chunk(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
##
def get_or_none(lst, n):
    if n in lst:
        return lst[n]
    else:
        return None
##
