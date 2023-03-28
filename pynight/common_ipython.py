import IPython
import sys
from .common_redirections import fd_redirected


def embed_stdin_to_stderr(*args, **kwargs):
    with fd_redirected(to=sys.stderr, original=sys.stdin, open_mode="rb"):
        return IPython.embed(*args, **kwargs)
