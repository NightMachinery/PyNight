from types import SimpleNamespace

import dns.resolver
import pytest

from pynight.common_networking import my_ip_get


def txt(text):
    return SimpleNamespace(strings=(text.encode(),))


@pytest.mark.parametrize(
    "records, expected",
    [
        ([txt("203.0.113.7")], "203.0.113.7"),
        #: What a recursive resolver adds; never taken for the address.
        ([txt("edns0-client-subnet 203.0.113.0/24"), txt("2001:db8::7")], "2001:db8::7"),
        ([txt("edns0-client-subnet 203.0.113.0/24")], ""),
    ],
)
def test_my_ip_get(monkeypatch, records, expected):
    asked = []
    monkeypatch.setattr(
        dns.resolver, "resolve",
        lambda name, rdtype, **kw: [SimpleNamespace(address="192.0.2.53")],
    )

    def resolve(self, name, rdtype, **kw):
        asked.append((tuple(self.nameservers), name, rdtype))
        return records

    monkeypatch.setattr(dns.resolver.Resolver, "resolve", resolve)
    assert my_ip_get() == expected
    assert asked == [(("192.0.2.53",), "o-o.myaddr.l.google.com", "TXT")]
