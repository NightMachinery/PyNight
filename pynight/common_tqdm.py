from tqdm.contrib.telegram import TelegramIO
from tqdm.auto import tqdm as tqdm_auto
from os import getenv


##
class tqdm_telegram(tqdm_auto):
    """
    Forked from `tqdm.contrib.telegram`.

    Standard `tqdm.auto.tqdm` but also sends updates to a Telegram Bot.
    May take a few seconds to create (`__init__`).

    - create a bot <https://core.telegram.org/bots#6-botfather>
    - copy its `{token}`
    - add the bot to a chat and send it a message such as `/start`
    - go to <https://api.telegram.org/bot`{token}`/getUpdates> to find out
      the `{chat_id}`
    - paste the `{token}` & `{chat_id}` below

    >>> from tqdm.contrib.telegram import tqdm, trange
    >>> for i in tqdm(iterable, token='{token}', chat_id='{chat_id}'):
    ...     ...
    """

    def __init__(
        self,
        *args,
        super_enabled_p="auto",
        tlg_enabled_p=True,
        leave_telegram=True,
        mininterval=1.0,
        ascii=True,
        **kwargs
    ):
        """
        Parameters
        ----------
        token  : str, required. Telegram token
            [default: ${TQDM_TELEGRAM_TOKEN}].
        chat_id  : str, required. Telegram chat ID
            [default: ${TQDM_TELEGRAM_CHAT_ID}].
        leave_telegram : bool. Whether to leave the message be after termination. Analogous to the normal `leave` argument.

        See `tqdm.auto.tqdm.__init__` for other parameters.
        """
        self.leave_telegram = leave_telegram

        kwargs["mininterval"] = mininterval
        kwargs["ascii"] = ascii

        token = kwargs.pop("token", getenv("TQDM_TELEGRAM_TOKEN"))
        chat_id = kwargs.pop("chat_id", getenv("TQDM_TELEGRAM_CHAT_ID"))
        if not (token and chat_id):
            tlg_enabled_p = False
        self.tlg_enabled_p = tlg_enabled_p

        if super_enabled_p == "auto":
            super_enabled_p = not self.tlg_enabled_p
        self.super_enabled_p = super_enabled_p

        if self.tlg_enabled_p:
            # if not kwargs.get("disable"):
            # kwargs = kwargs.copy()
            self.tgio = TelegramIO(
                token,
                chat_id,
            )

        super(tqdm_telegram, self).__init__(*args, **kwargs)

    def display(self, **kwargs):
        if self.super_enabled_p:
            super(tqdm_telegram, self).display(**kwargs)

        if self.tlg_enabled_p:
            fmt = self.format_dict
            if fmt.get("bar_format", None):
                fmt["bar_format"] = (
                    fmt["bar_format"]
                    .replace("<bar/>", "{bar:10u}")
                    .replace("{bar}", "{bar:10u}")
                )
            else:
                fmt["bar_format"] = "{l_bar}{bar:10u}{r_bar}"
            self.tgio.write(self.format_meter(**fmt))

    def clear(self, *args, **kwargs):
        super(tqdm_telegram, self).clear(*args, **kwargs)

        if self.tlg_enabled_p:
            # if not self.disable:
            self.tgio.write("")

    def close(self):
        if self.disable:
            return

        super(tqdm_telegram, self).close()

        if self.tlg_enabled_p and not self.leave_telegram:
            self.tgio.delete()


def ttgrange(*args, **kwargs):
    return tqdm_telegram(range(*args), **kwargs)


#: @aliases
tqdm = tqdm_telegram
trange = ttgrange
##
