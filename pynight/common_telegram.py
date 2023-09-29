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
import threading
from threading import Thread, Condition
from pynight.common_iterable import to_iterable
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

class AtomicCounter:
    def __init__(self):
        self.value = 0
        self._lock = threading.Lock()

    def increment(self):
        with self._lock:
            self.value += 1
            return self.value

global_order_index = AtomicCounter()

def send(
    *args,
    wait_p=False,
    **kwargs,
):
    order_index = global_order_index.increment()

    if wait_p:
        return _send(*args, wait_p=wait_p, order_index=order_index, **kwargs)
    else:
        return thread_pool.submit(_send, *args, wait_p=wait_p, order_index=order_index, **kwargs)


def _send(
    chat_id,
    files=None,
    msg="",
    wait_p=False,
    savefig_opts=None,
    lock_path=None,
    lock_key=None,
    autobatch=False,
    album_p=True,
    order_index=0,
    parse_mode=None,
):
    chat_id = chat_id or os.environ.get("tlogs", None)
    savefig_opts = savefig_opts or dict()

    files = to_iterable(files)

    file_paths = []
    with Timed(name="telegram_process_files", enabled_p=False):
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
        return thread_pool.submit(
            send_autobatch, file_paths=file_paths, order_index=order_index, chat_id=chat_id, msg=msg
        )

    cmd = [
        "tsend.py",
    ]
    for file_path in file_paths:
        cmd += [
            "--file",
            file_path,
        ]

    if album_p:
        cmd += ["--album"]
    else:
        cmd += ["--no-album"]

    if parse_mode:
        cmd += ["--parse-mode", parse_mode]

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
            _run_cmd(cmd)
        else:
            subprocess.Popen(cmd)


def _run_cmd(cmd):
    # ic(cmd)

    subprocess.check_call(cmd)


##
# A queue for each unique pair of (msg, chat_id)
autobatch_queues = defaultdict(list)

# A condition variable for each unique pair of (msg, chat_id)
autobatch_conditions = defaultdict(Condition)

# A separate thread for each unique pair of (msg, chat_id)
autobatch_threads = defaultdict(Thread)


def send_autobatch(chat_id, file_paths, msg="", order_index=0):
    # condition_key = chat_id
    condition_key = (msg, chat_id)

    file_ipaths = [(order_index, f) for f in file_paths]

    with autobatch_conditions[condition_key]:
        autobatch_queues[(msg, chat_id)].extend(file_ipaths)

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

            order_indices = [f[0] for f in files]
            files.sort(
                key=(lambda tup: tup[0]), #: order_index
            ) #: inplace, stable
            #: I have checked, this sorting is absolutely needed.

            files = [f[1] for f in files] #: getting the paths


            # if True:
            if False:
                # ic(order_indices)
                send(
                    msg=f"len={len(order_indices)}, {str(order_indices)}",
                    chat_id=chat_id,
                    wait_p=True,
                    lock_key=chat_id,
                    album_p=False,
                )

            send(
                files=files,
                msg=msg,
                # msg=msg,
                chat_id=chat_id,
                wait_p=True,
                lock_key=chat_id,
                album_p=False,
            )


##
