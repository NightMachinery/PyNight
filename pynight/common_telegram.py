from pathlib import Path
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import tempfile
import subprocess
import os
from brish import zn


##
def log_tlg(message, chat_id=None):
    chat_id = chat_id or os.environ.get("tlogs", None)
    return zn("tsend -- {chat_id} {message}")


##
def send_file(file, chat_id, msg="", wait_p=False, savefig_opts=None, lock_path=None):
    chat_id = chat_id or os.environ.get("tlogs", None)
    savefig_opts = savefig_opts or dict()

    # Handle case if file is a matplotlib plot
    if isinstance(file, Figure):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
            file.savefig(temp.name, format="png", **kwargs)
            file_path = temp.name
    # If file is a string or a Path object
    elif isinstance(file, (str, Path)):
        file_path = str(file)
    else:
        raise ValueError(f"Unsupported type: {type(file)}")

    cmd = [
        "tsend.py",
        "--file",
        file_path,
    ]
    if lock_path:
        cmd += [
            "--lock-path",
            lock_path,
        ]

    cmd += [
        "--",
        str(chat_id),
        msg,
    ]

    # If we need to wait for the command to finish
    if wait_p:
        subprocess.check_call(cmd)
    else:
        subprocess.Popen(cmd)


##
