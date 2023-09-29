##
def try_float(value):
    try:
        return float(value)
    except ValueError:
        return value


##
