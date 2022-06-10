from .common_networking import my_ip_get
from .common_telegram import log_tlg
from pydantic import BaseSettings
import traceback
import logging
from fastapi import Request


class FastAPISettings(BaseSettings):
    # disabling the docs
    openapi_url: str = ""  # "/openapi.json"


###
def request_path_get(request: Request):
    return request.scope.get("path", "")


##
class EndpointLoggingFilter1(logging.Filter):
    def __init__(
        self, *args, isDbg=False, logger=None, skip_paths=(), **kwargs
    ):
        self.isDbg = isDbg
        self.logger = logger
        self.skip_paths = skip_paths
        super().__init__(*args, **kwargs)

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            if self.isDbg:
                # self.logger and self.logger.info(f"LogRecord:\n{record.__dict__}")
                return True

            if hasattr(record, "scope"):
                req_path = record.scope.get("path", "")
            else:
                req_path = record.args[2]

            return not (req_path in self.skip_paths)
        except:
            res = traceback.format_exc()
            try:
                res += f"\n\nLogRecord:\n{record.__dict__}"
                ##
                msg: str = record.getMessage()
                res += f"\n\nmsg:\n{msg}"
                res += f"\n{msg.__dict__}"
            except:
                pass

            if self.logger:
                self.logger.warning(res)

            return True


###
seenIPs = None


def seenIPs_init():
    global seenIPs

    seenIPs = {"127.0.0.1", my_ip_get()}


def check_ip(request: Request, logger=None):
    if not seenIPs:
        seenIPs_init()

    first_seen = False
    ip = request.client.host
    if not (ip in seenIPs):
        first_seen = True
        logger and logger.warning(f"New IP seen: {ip}")
        # We log the IP separately, to be sure that an injection attack can't stop the message.
        log_tlg(f"New IP seen by the Garden: {ip}")
        seenIPs.add(ip)

    return ip, first_seen


###
