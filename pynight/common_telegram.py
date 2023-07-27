from typing import Iterable
from pathlib import Path
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import tempfile
import subprocess
import os
from brish import zn
from pathlib import Path
import tempfile
from collections import defaultdict
import concurrent.futures
from pynight.common_icecream import ic


##
def log_tlg(message, chat_id=None):
    chat_id = chat_id or os.environ.get("tlogs", None)
    return zn("tsend -- {chat_id} {message}")


##
#: Create a ThreadPoolExecutor with a single worker thread for each lock_key
lock_key_executors = defaultdict(
    lambda: concurrent.futures.ThreadPoolExecutor(max_workers=1)
)


def send(
    chat_id,
    files=None,
    msg="",
    wait_p=False,
    savefig_opts=None,
    lock_path=None,
    lock_key=None,
):
    chat_id = chat_id or os.environ.get("tlogs", None)
    savefig_opts = savefig_opts or dict()

    cmd = [
        "tsend.py",
    ]

    if files is None:
        files = []
    if isinstance(files, str) or not isinstance(files, Iterable):
        files = [files]

    for file in files:
        # Handle case if file is a matplotlib plot
        if isinstance(file, Figure):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
                file.savefig(temp.name, format="png", **savefig_opts)
                file_path = temp.name
        # If file is a string or a Path object
        elif isinstance(file, (str, Path)):
            file_path = str(file)
        else:
            raise ValueError(f"Unsupported type: {type(file)}")

        cmd += [
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

    if lock_key is not None:
        # Use the executor associated with this lock_key to run the command
        executor = lock_key_executors[lock_key]

        future = executor.submit(_run_cmd, cmd)
        if wait_p:
            return future.result()
        else:
            # ic(future, cmd)
            return future
    else:
        if wait_p:
            subprocess.check_call(cmd)
        else:
            subprocess.Popen(cmd)


def _run_cmd(cmd):
    subprocess.check_call(cmd)


##
