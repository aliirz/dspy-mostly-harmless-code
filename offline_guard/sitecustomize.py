"""Refuse outbound network connections inside an offline test run.

Placed on ``PYTHONPATH`` by ``test_offline_policy.py``; CPython imports a
module named ``sitecustomize`` automatically at interpreter start-up, before
any test does its work. Any socket that would leave the machine raises, so an
offline run that regressed into a provider call fails loudly instead of
spending money. AF_UNIX sockets stay usable (local IPC, not the network).
"""

import socket

_ALLOWED_FAMILIES = {getattr(socket, "AF_UNIX", None)}

BLOCKED = []


def _block(what, target=None):
    BLOCKED.append(f"{what}:{target}")
    raise RuntimeError(
        f"offline test run blocked a network call ({what} {target!r}); "
        "run with a provider key and without SKIP_LIVE_TESTS for live tests"
    )


_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex
_real_create_connection = socket.create_connection
_real_getaddrinfo = socket.getaddrinfo


def _connect(self, address, *args, **kwargs):
    if self.family in _ALLOWED_FAMILIES:
        return _real_connect(self, address, *args, **kwargs)
    _block("socket.connect", address)


def _connect_ex(self, address, *args, **kwargs):
    if self.family in _ALLOWED_FAMILIES:
        return _real_connect_ex(self, address, *args, **kwargs)
    _block("socket.connect_ex", address)


def _create_connection(address, *args, **kwargs):
    _block("socket.create_connection", address)


def _getaddrinfo(host, port, *args, **kwargs):
    _block("socket.getaddrinfo", host)


socket.socket.connect = _connect
socket.socket.connect_ex = _connect_ex
socket.create_connection = _create_connection
socket.getaddrinfo = _getaddrinfo
