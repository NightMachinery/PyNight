import time
import sys
from functools import wraps
from pynight.common_functional import fn_name


##
def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(
            f"\nTime: {fn_name(func)}: {end - start} seconds",
            flush=True,
            file=sys.stderr,
        )
        return result

    return wrapper


##


class Timed:
    def __init__(self, name="", enabled_p=True, print_p=True, output_dict=None):
        self.name = name
        self.enabled = enabled_p
        self.print_p = print_p
        self.output_dict = output_dict

    def __enter__(self):
        if self.enabled:
            self.start = time.time()

    def __exit__(self, type, value, traceback):
        if self.enabled:
            end = time.time()

            time_taken = end - self.start
            if self.output_dict is not None:
                self.output_dict["time"] = time_taken

            if self.print_p:
                print(
                    f"\nTime: {self.name}: {time_taken} seconds",
                    flush=True,
                    file=sys.stderr,
                )


##
