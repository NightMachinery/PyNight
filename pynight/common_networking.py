from brish import brishzn
from .common_debugging import traceback_print
from requests import get
import dns.resolver
import re


def my_ip_get():
    """This host's public IP address, or "" when it cannot be found.

    Asks Google's authoritative name server for the TXT record of
    o-o.myaddr.l.google.com, which holds the address the query came from.
    A recursive resolver such as 8.8.4.4 answers instead with its own
    address and, in a second record in no fixed order, the client's /24
    subnet ("edns0-client-subnet x.x.x.0/24"), never the client's address.
    """
    ip = ""
    try:
        # BrishGarden uses this function, so we can't call BrishGarden here :))
        # ip = brishzn(["myip"]).outrs
        ##
        resolver = dns.resolver.Resolver(configure=False)
        resolver.nameservers = [
            a.address for a in dns.resolver.resolve("ns1.google.com", "A", lifetime=3)
        ]
        answer = resolver.resolve("o-o.myaddr.l.google.com", "TXT", lifetime=3)
        for record in answer:
            text = b"".join(record.strings).decode("ascii")
            if re.fullmatch(r"[0-9.]+|[0-9a-fA-F:]+", text):
                ip = text
                break

        ##
        # ip = get('https://api.ipify.org').content.decode('utf8')

        # ip = get('https://ident.me').content.decode('utf8')
        ##
    except:
        traceback_print()
        pass

    return ip


##
