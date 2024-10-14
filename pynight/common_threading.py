import threading


SENTINEL_UNSET = "MAGIC_SENTINEL_UNSET_1vdang90Eg7Mi5D"


class ThreadWithResult(threading.Thread):
    #: @untested @o1
    ##
    def __init__(self, *args, **kwargs):
        threading.Thread.__init__(self, *args, **kwargs)
        self.result = SENTINEL_UNSET

    def run(self):
        self.result = SENTINEL_UNSET

        if self._target:
            #: Store the result of the target function
            self.result = self._target(*self._args, **self._kwargs)
