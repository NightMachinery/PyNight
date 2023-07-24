import time
import os
import asyncio
import aiofile
import tempfile
from pathlib import Path
from pynight.common_dict import simple_obj
from pynight.common_files import (
    mkdir,
    rm,
)


##
def lock_path_generate(
    *,
    lock_name=None,
    lock_path=None,
):
    if not lock_path and lock_name:
        lock_path = f"{Path.home()}/.locks/{lock_name}.lock"
    assert lock_path, "lock_path or lock_name must be supplied"

    return lock_path


async def lock_acquire(
    *,
    lock_name=None,
    lock_path=None,
    timeout=None,
    verbose_p=False,
    force_after_timeout_p=False,
    sleep_duration=1,
):
    lock_path = lock_path_generate(
        lock_name=lock_name,
        lock_path=lock_path,
    )
    mkdir(lock_path, do_dirname=True)
    success_p = False
    start_time = time.time()

    async def write_to_file(text, mode):
        async with aiofile.AIOFile(lock_path, mode) as lock:
            await lock.write(text)

    while True:
        try:
            await write_to_file(str(os.getpid()), "x")
            success_p = True
            if verbose_p:
                print(f"Lock {lock_name} acquired")
            break

        except FileExistsError:
            if timeout is not None and time.time() - start_time > timeout:
                if force_after_timeout_p:
                    await write_to_file(str(os.getpid()), "w")
                    success_p = True
                    if verbose_p:
                        print(f"Lock {lock_name} acquired by force after timeout")
                    break
                else:
                    if verbose_p:
                        print(
                            f"Failed to acquire lock {lock_name} within the specified timeout"
                        )
                    break
            else:
                if verbose_p:
                    print(
                        f"Lock {lock_name} is currently held by another process, waiting..."
                    )
                await asyncio.sleep(sleep_duration)

    return simple_obj(
        lock_path=lock_path,
        success_p=success_p,
    )


async def lock_release(
    *, lock_name=None, lock_path=None, check_pid_p=True, verbose_p=False
):
    lock_path = lock_path_generate(
        lock_name=lock_name,
        lock_path=lock_path,
    )

    if os.path.exists(lock_file):
        if check_pid_p:
            async with aiofile.AIOFile(lock_file, "r") as lock:
                pid = await lock.read()
                if pid != str(os.getpid()):
                    if verbose_p:
                        print(
                            f"Cannot release lock {lock_name} held by another process"
                        )
                    return False

        os.remove(lock_file)
        if verbose_p:
            print(f"Lock {lock_name} released")
        return True
    else:
        if verbose_p:
            print(f"No lock {lock_name} to release")
    return False


##
