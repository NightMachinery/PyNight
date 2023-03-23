import openai
import os
from brish import z, zp

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

def chatml_response_process(response, copy_mode='default'):
    for choice in response["choices"]:
        text = choice["message"]["content"]
        text_m = f"""        {{"role": "assistant", "content": '''{text}'''}},"""

        if False:
            text += """
        {{"role": "user", "content": ''' '''}},"""

        if copy_mode == 'default':
            pyperclip.copy(text)
            pyperclip.copy(text_m)
        elif copy_mode == 'text':
            pyperclip.copy(text)

        print(text)
        print("-------")


def writegpt_process(messages_lst):
    out = ""
    seen = ["PLACEHOLDER",]
    #: We can also just count the number of assistant outputs previously seen, and skip exactly that many. That way, we can edit the text more easily.

    for messages in messages_lst:
        for message in messages:
            role = message["role"]
            content = message["content"]

            if role in ("assistant",) and content not in seen :
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

def openai_chat_complete(*args, **kwargs):
    while True:
        try:
            return openai.ChatCompletion.create(*args, **kwargs)
        except openai.error.RateLimitError:
            print("OpenAI ratelimit encountered, sleeping ...", file=sys.stderr)
            time.sleep(10) #: in seconds
            pass


###
