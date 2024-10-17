# from pynight.common_debugging import *
import threading
import re
import sys
import traceback
import os
from functools import wraps
from pynight.common_condition import jupyter_p
from pynight.common_hosts import hostname_get
from pynight.common_telegram import (
    tlg_chat_id_default,
)
from pynight.common_telegram import send as tlg_send
from pynight.common_bells import (
    bell_call_remote,
)
from brish import z
from IPython.core import ultratb

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


##
def ipdb_enable(
    tlg_chat_id=tlg_chat_id_default,  #: Use None to disable
    torch_oom_mode="no_pdb",
    ##
    non_interactive_exceptions="auto",
    # non_interactive_exceptions=None,
    non_interactive_base_exception_p=True,
    # non_interactive_traceback_mode="Verbose",
    non_interactive_traceback_mode="Context",
    ##
    jupyter_mode=True,
    bell_name="bell-python-error",
    jupyter_bell_name="bell-jupyter-error",
):
    in_jupyter_p = jupyter_p()
    #: jupyter_p returns False for IPython terminal shells

    if in_jupyter_p:
        if jupyter_mode == "disabled":
            return

        non_interactive_traceback_mode = "Context"
        #: Verbose can be too verbose and crash emacs
        #: I am using `print_traceback()` instead.

    if non_interactive_exceptions is None:
        non_interactive_exceptions = []

    elif non_interactive_exceptions == "auto":
        from urllib.error import URLError, HTTPError
        import socket

        non_interactive_exceptions = [
            URLError,
            HTTPError,
            socket.timeout,
            TimeoutError,
            ConnectionError,
            OSError,
        ]

    #: [[https://ipython.readthedocs.io/en/8.18.0/api/generated/IPython.core.ultratb.html][Module: core.ultratb — IPython 8.18.0 documentation]]
    pdb_excepthook = ultratb.FormattedTB(
        mode="Context",
        color_scheme="Linux",
        call_pdb=1,
    )
    non_pdb_excepthook = ultratb.FormattedTB(
        mode=non_interactive_traceback_mode,
        color_scheme="Linux",
        call_pdb=0,
    )

    @wraps(sys.excepthook)
    def excepthook_wrapper(exc_type, exc_value, exc_traceback):
        def print_traceback():
            #: @duplicateCode/be81cfe1eb287b53e25e5d62d3a838a5
            ##
            try:
                print(traceback.format_exc(), file=sys.stderr)

            except Exception:
                pass

        if tlg_chat_id:

            #: Send notification to Telegram in a separate thread
            def send_telegram_notification():
                try:
                    error_message = f"Error: {exc_type.__name__}: {str(exc_value)}"
                    msg = f"@{hostname_get()}\n{error_message}"

                    tlg_send(
                        chat_id=tlg_chat_id,
                        msg=msg,
                        parse_mode="none",
                        wait_p=True,
                        #: We are running this in another thread anyway.
                    )

                except Exception as e:
                    print(f"Error sending Telegram notification:\n  {e}")

            threading.Thread(target=send_telegram_notification, daemon=True).start()

        if in_jupyter_p:
            bell_call_remote(jupyter_bell_name)

            print_traceback()

            return non_pdb_excepthook(exc_type, exc_value, exc_traceback)

        else:
            bell_call_remote(bell_name)

            if TORCH_AVAILABLE and isinstance(exc_value, torch.cuda.OutOfMemoryError):
                print("CUDA OutOfMemoryError occurred.", file=sys.stderr)
                if torch_oom_mode == "no_pdb":
                    print_traceback()
                    return non_pdb_excepthook(exc_type, exc_value, exc_traceback)

            elif (
                non_interactive_base_exception_p and not issubclass(exc_type, Exception)
            ) or (
                any(isinstance(exc_value, exc) for exc in non_interactive_exceptions)
            ):
                print_traceback()
                return non_pdb_excepthook(exc_type, exc_value, exc_traceback)

            else:
                return pdb_excepthook(exc_type, exc_value, exc_traceback)

    sys.excepthook = excepthook_wrapper
    # print(
    #     f"hooked sys.excepthook: {sys.excepthook}",
    #     file=sys.stderr,
    # )

    if in_jupyter_p:
        from IPython.core.interactiveshell import InteractiveShell

        def custom_exc(shell, etype, evalue, tb, tb_offset=None):
            excepthook_wrapper(etype, evalue, tb)

        shell = InteractiveShell.instance()
        shell.set_custom_exc((BaseException,), custom_exc)


##
