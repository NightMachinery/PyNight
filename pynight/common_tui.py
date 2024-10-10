import sys


##
def prompt_user(question, end="\n> "):
    """Prompt the user for input using /dev/tty if available, else use standard input."""

    question = question + end
    try:
        with open("/dev/tty", "w") as tty_out, open("/dev/tty", "r") as tty_in:
            tty_out.write(question)
            tty_out.flush()  # Ensure the question is displayed before reading input
            return tty_in.readline().strip().lower()

    except (FileNotFoundError, IOError):
        return input(question).strip().lower()


def ask(question, default=True):
    """Ask a yes/no question and return the answer as True or False.

    If the user presses enter without providing an answer, return the default value unless default is None.
    """
    if default is not None:
        if default:
            question += " [Y/n]"
        else:
            question += " [y/N]"

    else:
        question += " [y/n]"

    while True:
        answer = prompt_user(question)

        #: If the user just presses enter, return the default value if provided
        if not answer and default is not None:
            return default

        elif answer in ["y", "yes"]:
            return True

        elif answer in ["n", "no"]:
            return False

        else:
            print("Please answer with 'y'/'n'.")


##
