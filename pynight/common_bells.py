from brish import (
    z,
    z_background,
)


def bello():
    z("bello")


def bell_gpt():
    z("bell-gpt")


def bell_call_remote(bell_name):
    if bell_name:
        z_background("bell-call-remote {bell_name}")
