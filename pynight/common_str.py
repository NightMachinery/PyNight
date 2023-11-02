import sys


##
def print_to_file(*args, file=sys.stdout, **kwargs):
    if isinstance(file, str):
        file_obj = open(file, 'a')
    else:
        file_obj = file

    try:
        print(*args, file=file_obj, **kwargs)
    finally:
        if isinstance(file, str):
            file_obj.close()
##
def try_float(value):
    try:
        return float(value)
    except ValueError:
        return value


##
