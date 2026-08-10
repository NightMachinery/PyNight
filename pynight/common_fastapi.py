from .common_apikey import API_KEY_HEADER_NAME, api_key_ensure
from .common_networking import my_ip_get
from .common_telegram import log_tlg

# from pydantic import BaseSettings
from pydantic_settings import BaseSettings

#: pip install pydantic-settings

import secrets
import traceback
import logging
from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader


class FastAPISettings(BaseSettings):
    # disabling the docs
    openapi_url: str = ""  # "/openapi.json"


###
def request_path_get(request: Request):
    return request.scope.get("path", "")


##
class EndpointLoggingFilter1(logging.Filter):
    def __init__(self, *args, isDbg=False, logger=None, skip_paths=(), **kwargs):
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
def api_key_dependency_make(service_name, logger=None):
    """Build an app-level dependency that requires the service's API key.

    Pass it as `FastAPI(dependencies=[Depends(api_key_dependency_make(...))])`
    so it guards every route and runs before the endpoint body; an endpoint that
    swallows exceptions can then never swallow the 401.

    Binding loopback keeps other hosts out, but not other users of this host,
    nor a browser tricked into POSTing to 127.0.0.1. Requiring a custom header
    additionally forces a CORS preflight, which such a browser cannot pass.
    """

    expected_key = api_key_ensure(service_name)
    header_scheme = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)

    def api_key_verify(request: Request, api_key: str = Security(header_scheme)):
        if not (api_key and secrets.compare_digest(api_key, expected_key)):
            logger and logger.warning(
                f"Rejected a request with a missing or invalid {API_KEY_HEADER_NAME}:"
                f" ip={request.client.host} path={request_path_get(request)}"
            )
            raise HTTPException(
                status_code=401, detail=f"Invalid or missing {API_KEY_HEADER_NAME}"
            )

    return api_key_verify


###
