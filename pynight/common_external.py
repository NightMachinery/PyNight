from brish import CmdResult
import subprocess
import os

##
def html2org(html):
    cmd_array = [
        "pandoc",
        "--wrap=none",
        "--from",
        "html",
        "--to",
        "org",
        "-",
        "-o",
        "-",
    ]

    sp = subprocess.run(
        cmd_array,
        shell=False,
        cwd=os.getcwd(),
        text=True,
        executable=None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        input=html,
    )

    return CmdResult(sp.returncode, sp.stdout, sp.stderr, cmd_array, html)


##
