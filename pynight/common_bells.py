from brish import (
    z,
    z_background,
)


def bello():
    z("bello")


def bell_gpt():
    z("bell-gpt")


def bell_call_remote(bell_name):
    try:
        if bell_name:
            z_background("bell-call-remote {bell_name}")

    except Exception as e:
        print(f"Error remote-calling bell:\n  bell_name: {bell_name}\n  {e}")
