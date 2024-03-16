import anthropic
from brish import z, zp


anthropic_client = None
##
anthropic_key = None


def setup_anthropic_key():
    global anthropic_key
    global anthropic_client

    anthropic_key = z("var-get anthropic_api_key").outrs
    assert anthropic_key, "setup_anthropic_key: could not get Anthropic API key!"

    anthropic_client = anthropic.Anthropic(
        # defaults to os.environ.get("ANTHROPIC_API_KEY")
        api_key=anthropic_key,
    )
    return anthropic_client


def anthropic_key_get():
    global anthropic_key

    if anthropic_key is None:
        setup_anthropic_key()

    return anthropic_key


##
