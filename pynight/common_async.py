import functools
import asyncio
from concurrent.futures import ThreadPoolExecutor

##
def force_async(f):
    @functools.wraps(f)
    def inner(*args, **kwargs):
        loop = asyncio.get_running_loop()
        return loop.run_in_executor(None, lambda: f(*args, **kwargs))

    return inner


##
def async_max_workers_set(n):
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=(n))
    loop.set_default_executor(executor)
    return executor


##
