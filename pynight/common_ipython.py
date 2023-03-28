import IPython
import sys
from .common_redirections import fd_redirected


def embed_tty(*args1872782, locals_=None, **kwargs23828237):
    if locals_ is None:
        previous_frame = sys._getframe(1)
        previous_frame_locals = previous_frame.f_locals
        locals_ = dict(previous_frame.f_globals, **previous_frame_locals)

    locals_["args1872782"] = args1872782
    locals_["kwargs23828237"] = kwargs23828237
    locals_["IPython2882872827"] = IPython

    with open('/dev/tty') as user_tty:
        stdin_orig = sys.stdin
        try:
            sys.stdin = user_tty
            return exec("IPython2882872827.embed(*args1872782, **kwargs23828237)", locals_)
        finally:
            sys.stdin = stdin_orig


def embed_stdin_to_stderr(*args1872782, locals_=None, **kwargs23828237):
    if locals_ is None:
        previous_frame = sys._getframe(1)
        previous_frame_locals = previous_frame.f_locals
        locals_ = dict(previous_frame.f_globals, **previous_frame_locals)

    locals_["args1872782"] = args1872782
    locals_["kwargs23828237"] = kwargs23828237
    locals_["IPython2882872827"] = IPython

    with fd_redirected(to=sys.stderr, original=sys.stdin, open_mode="rb"):
        return exec("IPython2882872827.embed(*args1872782, **kwargs23828237)", locals_)
