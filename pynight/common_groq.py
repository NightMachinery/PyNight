import groq

from brish import z, zp


groq_client = None
##
groq_key = None


def setup_groq_key(*, raise_error_p=False):
    global groq_key
    global groq_client

    groq_key = z("var-get groq_api_key").outrs
    if not groq_key:
        if raise_error_p:
            assert False, "setup_groq_key: could not get Groq API key!"
        else:
            return None

    groq_client = groq.Groq(
        # defaults to os.environ.get("GROQ_API_KEY")
        api_key=groq_key,
    )
    return groq_client


def groq_key_get():
    global groq_key

    if groq_key is None:
        setup_groq_key()

    return groq_key


##
