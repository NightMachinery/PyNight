import pyperclip
import threading
import time


def clipboard_copy(obj):
    return pyperclip.copy(obj)


def clipboard_copy_multi_sync(
    texts,
    sleep=0.5,
):
    for text in texts:
        if text:  #: Check if the text is not empty or None
            clipboard_copy(text)
            time.sleep(sleep)
            #: to allow polling-based clipboard managers to capture the text


def clipboard_copy_multi(*args):
    copy_thread = threading.Thread(target=clipboard_copy_multi_sync, args=(args,))

    copy_thread.start()
