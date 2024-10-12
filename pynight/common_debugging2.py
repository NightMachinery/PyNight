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
from brish import z
from IPython.core import ultratb

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


##
def ipdb_enable(
    disable_in_jupyter_p=True,
    tlg_chat_id=tlg_chat_id_default,  #: Use None to disable
    torch_oom_mode="no_pdb",
    non_interactive_exceptions="auto",
    # non_interactive_exceptions=None,
    non_interactive_base_exception_p=True,
):
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

    if disable_in_jupyter_p and jupyter_p():
        return

    pdb_excepthook = ultratb.FormattedTB(
        mode="Context",
        color_scheme="Linux",
        call_pdb=1,
    )
    non_pdb_excepthook = ultratb.FormattedTB(
        mode="Context",
        color_scheme="Linux",
        call_pdb=0,
    )

    @wraps(sys.excepthook)
    def excepthook_wrapper(exc_type, exc_value, exc_traceback):
        if tlg_chat_id:

            #: Send notification to Telegram in a separate thread
            def send_telegram_notification():
                try:
                    error_message = f"Error: {exc_type.__name__}: {str(exc_value)}"
                    msg = f"@{hostname_get()}\n{error_message}"

                    tlg_send(
                        chat_id=tlg_chat_id,
                        msg=msg,
                        wait_p=True,
                        #: We are running this in another thread anyway.
                    )

                except Exception as e:
                    print(f"Error sending Telegram notification:\n  {e}")

            threading.Thread(target=send_telegram_notification, daemon=True).start()

        if TORCH_AVAILABLE and isinstance(exc_value, torch.cuda.OutOfMemoryError):
            print("CUDA OutOfMemoryError occurred.", file=sys.stderr)
            if torch_oom_mode == "no_pdb":
                print("Not entering debugger.", file=sys.stderr)
                return non_pdb_excepthook(exc_type, exc_value, exc_traceback)

        if non_interactive_base_exception_p and not issubclass(exc_type, Exception):
            return non_pdb_excepthook(exc_type, exc_value, exc_traceback)

        if any(isinstance(exc_value, exc) for exc in non_interactive_exceptions):
            return non_pdb_excepthook(exc_type, exc_value, exc_traceback)

        return pdb_excepthook(exc_type, exc_value, exc_traceback)

    sys.excepthook = excepthook_wrapper


##
