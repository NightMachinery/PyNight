import time
from functools import wraps
from pynight.common_functional import fn_name


##
def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Time taken by {fn_name(func)}: {end - start} seconds")
        return result

    return wrapper


##


class Timed:
    def __init__(self, name="", enabled_p=True):
        self.name = name
        self.enabled = enabled_p

    def __enter__(self):
        if self.enabled:
            self.start = time.time()

    def __exit__(self, type, value, traceback):
        if self.enabled:
            end = time.time()
            print(f"Time taken by {self.name}: {end - self.start} seconds")


##
