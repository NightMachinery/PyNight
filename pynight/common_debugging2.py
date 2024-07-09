# from pynight.common_debugging import *
import threading
import re
import sys
import traceback
import os
from functools import wraps
from pynight.common_condition import jupyter_p
from pynight.common_hosts import hostname_get
from pynight.common_telegram import send as tlg_send
from brish import z
from IPython.core import ultratb


##
def ipdb_enable(
    disable_in_jupyter_p=True,
    tlg_chat_id="195391705",
):
    if disable_in_jupyter_p:
        if jupyter_p():
            return

    def telegram_notify_wrapper(original_excepthook):
        @wraps(original_excepthook)
        def wrapper(exc_type, exc_value, exc_traceback):
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

            return original_excepthook(exc_type, exc_value, exc_traceback)

        return wrapper

    original_excepthook = ultratb.FormattedTB(
        mode="Context",
        color_scheme="Linux",
        call_pdb=1,
    )

    if tlg_chat_id:
        sys.excepthook = telegram_notify_wrapper(original_excepthook)

    else:
        sys.excepthook = original_excepthook


##
