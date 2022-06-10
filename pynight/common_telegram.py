import os
from brish import zn


def log_tlg(message, chat=os.environ.get("tlogs")):
    return zn("tsend -- {chat} {message}")
