import os
import sys
from contextlib import contextmanager


def fileno(file_or_fd):
    fd = getattr(file_or_fd, "fileno", lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError(
            "Expected a file (`.fileno()`) or a file descriptor"
        )
    return fd


@contextmanager
def fd_redirected(to=os.devnull, original=None, open_mode="wb"):
    #: @forked from https://stackoverflow.com/questions/4675728/redirect-stdout-to-a-file-in-python
    #:
    #: * Redirect stdin:
    #:   =with fd_redirected(to=sys.stderr, original=sys.stdin, open_mode='rb'):=
    ##
    if original is None:
        original = sys.stdout

    original_fd = fileno(original)
    # copy original_fd before it is overwritten
    # NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(original_fd), open_mode) as copied:
        original.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), original_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, open_mode) as to_file:
                os.dup2(to_file.fileno(), original_fd)  # $ exec > to
        try:
            yield original  # allow code to be run with the redirected original
        finally:
            # restore original to its previous value
            # NOTE: dup2 makes original_fd inheritable unconditionally
            original.flush()
            os.dup2(copied.fileno(), original_fd)  # $ exec >&copied
##
