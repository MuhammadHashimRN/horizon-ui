"""Unit tests for IPC server."""

import pytest

from horizon.control.ipc_auth import IPCAuth
from horizon.control.ipc_server import IPCServer


class TestIPCServer:
    def test_register_method(self):
        auth = IPCAuth()
        server = IPCServer(auth=auth)
        server.register_method("test", lambda: "ok")
        assert "test" in server._handlers

    def test_make_response(self):
        resp = IPCServer._make_response(1, {"status": "ok"})
        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == 1
        assert resp["result"]["status"] == "ok"

    def test_make_error(self):
        resp = IPCServer._make_error(1, -32601, "Method not found")
        assert resp["error"]["code"] == -32601
