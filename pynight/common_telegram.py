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
from threading import Thread, Condition
from pynight.common_icecream import ic
from pynight.common_benchmark import (
    timed,
    Timed,
)
import time


##
def log_tlg(message, chat_id=None):
    chat_id = chat_id or os.environ.get("tlogs", None)
    return zn("tsend -- {chat_id} {message}")


##
#: Create a ThreadPoolExecutor with a single worker thread for each lock_key
lock_key_executors = defaultdict(
    lambda: concurrent.futures.ThreadPoolExecutor(max_workers=1)
)
thread_pool = concurrent.futures.ThreadPoolExecutor(max(os.cpu_count(), 5))

def send(
    *args,
    wait_p=False,
    **kwargs,
):
    if wait_p:
        return _send(*args, wait_p=wait_p, **kwargs)
    else:
        return thread_pool.submit(_send, *args, wait_p=wait_p, **kwargs)

def _send(
    chat_id,
    files=None,
    msg="",
    wait_p=False,
    savefig_opts=None,
    lock_path=None,
    lock_key=None,
    autobatch=False,
):
    chat_id = chat_id or os.environ.get("tlogs", None)
    savefig_opts = savefig_opts or dict()

    if files is None:
        files = []
    if isinstance(files, str) or not isinstance(files, Iterable):
        files = [files]

    file_paths = []
    with Timed(name='telegram_process_files', enabled_p=False):
    #: Time taken by telegram_process_files: 0.6532742977142334 seconds
    ##
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

            assert os.path.exists(file_path), f"Non-existent file_path: {file_path}"
            file_paths.append(file_path)

    if autobatch:
        #: Time taken by concurrent.futures.thread.ThreadPoolExecutor.submit: 4.57763671875e-05 seconds
        return thread_pool.submit(send_autobatch, file_paths=file_paths, chat_id=chat_id, msg=msg)

    cmd = [
        "tsend.py",
    ]
    for file_path in file_paths:
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
# A queue for each unique pair of (msg, chat_id)
autobatch_queues = defaultdict(list)

# A condition variable for each unique pair of (msg, chat_id)
autobatch_conditions = defaultdict(Condition)

# A separate thread for each unique pair of (msg, chat_id)
autobatch_threads = defaultdict(Thread)


def send_autobatch(chat_id, file_paths, msg=''):
    # condition_key = chat_id
    condition_key = (msg, chat_id)

    with autobatch_conditions[condition_key]:
        autobatch_queues[(msg, chat_id)].extend(file_paths)

        if not autobatch_threads[(msg, chat_id)].is_alive():
            autobatch_threads[(msg, chat_id)] = Thread(
                target=batch_sender, args=(msg, chat_id), daemon=True
            )
            autobatch_threads[(msg, chat_id)].start()
        else:
            autobatch_conditions[condition_key].notify_all()


def batch_sender(msg, chat_id, wait_ms=None):
    if wait_ms is None:
        # wait_ms = 1000
        wait_ms = 0
    # Convert wait_ms to seconds for time.sleep
    sleep_time = wait_ms / 1000

    condition_key = (msg, chat_id)

    while True:
        autobatch_conditions[condition_key].acquire()
        try:
            while not autobatch_queues[(msg, chat_id)]:
                autobatch_conditions[condition_key].wait()

        finally:
            autobatch_conditions[condition_key].release()

        if sleep_time:
            time.sleep(sleep_time)
        autobatch_conditions[condition_key].acquire()
        try:
            files = autobatch_queues[(msg, chat_id)].copy()
            del autobatch_queues[(msg, chat_id)][:]
        finally:
            autobatch_conditions[condition_key].release()

        if files:
            print(f"sending autobatched files (len={len(files)})")

            send(
                files=files,
                msg=msg,
                chat_id=chat_id,
                wait_p=True,
                lock_key=chat_id,
            )

##
