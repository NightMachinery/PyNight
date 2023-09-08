import sys

##
def prompt_user(question, end='\n'):
    """Prompt the user for input using /dev/tty if available, else use standard input."""

    try:
        with open('/dev/tty', 'r+') as tty:
            tty.write(question + end)
            tty.flush()  # Ensure the question is displayed before reading input
            return tty.readline().strip().lower()

    except (FileNotFoundError, IOError):
        return input(question).strip().lower()


def ask(question, default=True):
    """Ask a yes/no question and return the answer as True or False.

    If the user presses enter without providing an answer, return the default value.
    """
    # Modify the question to indicate the default
    if default:
        question += " [Y/n]"
    else:
        question += " [y/N]"

    while True:
        # Use the previously defined prompt_user function to get the answer
        answer = prompt_user(question)

        # If the user just presses enter, return the default value
        if not answer:
            return default
        # Interpret the user's answer
        elif answer in ["y", "yes"]:
            return True
        elif answer in ["n", "no"]:
            return False
        # If we get here, the user provided an invalid answer. Prompt again.
        else:
            print("Please answer with 'yes'/'no' (or 'y'/'n').")


##
