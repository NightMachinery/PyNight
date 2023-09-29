import tiktoken
from types import SimpleNamespace
import openai
import os
from brish import z, zp
from pynight.common_bells import bell_gpt
from pynight.common_dict import simple_obj

# openai.api_key = os.environ["OPENAI_API_KEY"]
openai.api_key = z('print -r -- "$openai_api_key"').outrs
#: 'openai_api_key' not actually exported

###
import openai
import pyperclip
from icecream import ic
import subprocess
import time
import sys


def chatml_response_text_process(
    text,
    copy_mode="default",
):
    #: 'rawchat' is currently useless, just use 'text'.
    ##
    text_m = None
    if copy_mode in ("chat", "chat2"):
        text_m = f"""        {{"role": "assistant", "content": '''{text}'''}},"""
    elif copy_mode in ("rawchat"):
        text_m = f"""{text}"""

    if copy_mode == "chat2":
        text_m += f"""
        {{"role": "user", "content": ''' '''}},"""

    if copy_mode == "default":
        pyperclip.copy(text)

        time.sleep(0.1)
        #: to allow polling-based clipboard managers to capture the text

        pyperclip.copy(text_m)

    elif copy_mode in ("chat", "chat2", "rawchat"):
        pyperclip.copy(text_m)

    elif copy_mode == "text":
        pyperclip.copy(text)

    return simple_obj(
        text=text,
        text_chat=text_m,
    )


def chatml_response_process(response, copy_mode="default"):
    for choice in response["choices"]:
        text = choice["message"]["content"]

        chatml_response_text_process(text)
        print(text)
        print("-------")


def writegpt_process(messages_lst):
    out = ""
    seen = [
        "PLACEHOLDER",
    ]
    #: We can also just count the number of assistant outputs previously seen, and skip exactly that many. That way, we can edit the text more easily.

    for messages in messages_lst:
        for message in messages:
            role = message["role"]
            content = message["content"]

            if role in ("assistant",) and content not in seen:
                seen.append(content)

                if out:
                    out += "\n\n"
                out += content

    out = subprocess.run(
        ["perl", "-CIOE", "-0777", "-pe", r"s/(*plb:\S)(\R)(*pla:\S)/\\\\$1/g"],
        text=True,
        input=out,
        errors="strict",
        encoding="utf-8",
        capture_output=True,
    ).stdout

    pyperclip.copy(out)
    return out


def openai_chat_complete(
    *args,
    model="gpt-3.5-turbo",
    messages=None,
    interactive=False,
    copy_last_message=None,
    bell=None,
    **kwargs,
):
    if interactive:
        if copy_last_message is None:
            copy_last_message = True
        if bell is None:
            bell = True

    try:
        while True:
            if copy_last_message:
                last_message = messages[-1]["content"]
                pyperclip.copy(last_message)

            try:
                return openai.ChatCompletion.create(
                    *args, model=model, messages=messages, **kwargs
                )
            except openai.error.RateLimitError:
                print(
                    "OpenAI ratelimit encountered, sleeping ...",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(10)  #: in seconds
    finally:
        if bell:
            bell_gpt()


###
def truncate_by_tokens(text, length=3500, model="gpt-3.5-turbo"):
    encoder = tiktoken.encoding_for_model(model)

    encoded = encoder.encode(text)

    truncate_p = len(encoded) > length
    encoded_rest = None
    if truncate_p:
        encoded_truncated = encoded[:length]
        encoded_rest = encoded[length:]
    else:
        encoded_truncated = encoded

    text_truncated = encoder.decode(encoded_truncated)
    text_rest = None
    if encoded_rest:
        text_rest = encoder.decode(encoded_rest)

    return SimpleNamespace(
        text=text_truncated,
        text_rest=text_rest,
        truncated_p=truncate_p,
    )


###
