import os
import secrets
from pathlib import Path

#: Key files live at `~/.keys/<service_name>` and hold a complete header line,
#: e.g. `X-API-Key: <key>`. That format lets shell clients attach the key with
#: `curl --header @~/.keys/<service_name>`, which never puts the secret in argv;
#: `ps` is readable by every local user, and local users are exactly the threat
#: this key defends against.
###
API_KEY_HEADER_NAME = "X-API-Key"
KEYS_DIR = Path.home() / ".keys"


def api_key_path_get(service_name):
    return KEYS_DIR / service_name


def api_key_read(service_name):
    """Return the bare key from `~/.keys/<service_name>`, or None if unavailable."""

    try:
        content = api_key_path_get(service_name).read_text().strip()
    except OSError:
        return None

    prefix = API_KEY_HEADER_NAME + ":"
    if content.lower().startswith(prefix.lower()):
        content = content[len(prefix) :].strip()

    return content or None


def api_key_ensure(service_name):
    """Return the service's key, generating its key file if missing or empty.

    Idempotent: an existing non-empty file always wins, so restarting a service
    never invalidates the keys its clients already hold.
    """

    key = api_key_read(service_name)
    if key is not None:
        return key

    KEYS_DIR.mkdir(mode=0o700, exist_ok=True)
    #: `mkdir(mode=...)` does not touch an already existing directory.
    KEYS_DIR.chmod(0o700)

    key = secrets.token_urlsafe(32)

    key_path = api_key_path_get(service_name)
    fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(f"{API_KEY_HEADER_NAME}: {key}\n")

    return key


###
