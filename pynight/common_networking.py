from brish import brishzn
from .common_debugging import traceback_print
from requests import get
import dns.resolver
import re


def my_ip_get():
    ip = ""
    try:
        # BrishGarden uses this function, so we can't call BrishGarden here :))
        # ip = brishzn(["myip"]).outrs
        ##
        resolver = dns.resolver.Resolver()
        resolver.nameservers = ["8.8.4.4"]
        answer = resolver.resolve("o-o.myaddr.l.google.com", "TXT")
        ip = re.match(
            r'"edns0-client-subnet (.*)/\d+"', str(answer.rrset[1])
        )[1]
        ##
        # ip = get('https://api.ipify.org').content.decode('utf8')

        # ip = get('https://ident.me').content.decode('utf8')
        ##
    except:
        traceback_print()
        pass

    return ip


##
