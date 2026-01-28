"""JSON-RPC 2.0 IPC server over Windows Named Pipes."""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Callable

from horizon.constants import MAX_PLUGIN_MESSAGE_SIZE, PIPE_NAME
from horizon.control.ipc_auth import IPCAuth

logger = logging.getLogger(__name__)


class IPCServer:
    """JSON-RPC 2.0 server using Windows Named Pipes.

    Provides an IPC endpoint for plugins to communicate with Horizon UI.
    Each connection is authenticated via token before processing messages.
    """

    def __init__(
        self,
        auth: IPCAuth,
        pipe_name: str = PIPE_NAME,
    ) -> None:
        self._auth = auth
        self._pipe_name = pipe_name
        self._handlers: dict[str, Callable[..., Any]] = {}
        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        logger.info("IPCServer initialized (pipe=%s)", pipe_name)

    def register_method(self, name: str, handler: Callable[..., Any]) -> None:
        self._handlers[name] = handler
        logger.debug("Registered RPC method: %s", name)

    def start(self) -> None:
        self._running.set()
        self._thread = threading.Thread(target=self._serve_loop, daemon=True, name="IPCServer")
        self._thread.start()
        logger.info("IPCServer started")

    def stop(self) -> None:
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=2)
        logger.info("IPCServer stopped")

    def _serve_loop(self) -> None:
        try:
            import win32pipe
            import win32file
            import pywintypes
        except ImportError:
            logger.error("pywin32 not installed, IPC server cannot start")
            return

        while self._running.is_set():
            try:
                pipe = win32pipe.CreateNamedPipe(
                    self._pipe_name,
                    win32pipe.PIPE_ACCESS_DUPLEX,
                    win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_READMODE_MESSAGE | win32pipe.PIPE_WAIT,
                    win32pipe.PIPE_UNLIMITED_INSTANCES,
                    MAX_PLUGIN_MESSAGE_SIZE,
                    MAX_PLUGIN_MESSAGE_SIZE,
                    0,
                    None,
                )

                # Wait for client connection
                try:
                    win32pipe.ConnectNamedPipe(pipe, None)
                except pywintypes.error:
                    win32file.CloseHandle(pipe)
                    continue

                # Handle connection in a separate thread
                threading.Thread(
                    target=self._handle_connection,
                    args=(pipe,),
                    daemon=True,
                ).start()

            except Exception:
                if self._running.is_set():
                    logger.exception("IPC server error")

    def _handle_connection(self, pipe) -> None:
        import win32file

        authenticated = False
        plugin_name: str | None = None

        try:
            while self._running.is_set():
                # Read message
                try:
                    _, data = win32file.ReadFile(pipe, MAX_PLUGIN_MESSAGE_SIZE)
                    message = json.loads(data.decode("utf-8"))
                except Exception:
                    break

                # First message must be authentication
                if not authenticated:
                    token = message.get("params", {}).get("token", "")
                    plugin_name = self._auth.validate_token(token)
                    if plugin_name:
                        authenticated = True
                        response = self._make_response(message.get("id"), {"status": "authenticated"})
                    else:
                        response = self._make_error(message.get("id"), -32000, "Authentication failed")
                        self._send_response(pipe, response)
                        break
                else:
                    response = self._process_request(message)

                self._send_response(pipe, response)

        except Exception:
            logger.exception("Connection handler error (plugin=%s)", plugin_name)
        finally:
            try:
                win32file.CloseHandle(pipe)
            except Exception:
                pass

    def _process_request(self, message: dict[str, Any]) -> dict[str, Any]:
        msg_id = message.get("id")
        method = message.get("method", "")
        params = message.get("params", {})

        handler = self._handlers.get(method)
        if not handler:
            return self._make_error(msg_id, -32601, f"Method not found: {method}")

        try:
            result = handler(**params) if isinstance(params, dict) else handler(*params)
            return self._make_response(msg_id, result)
        except Exception as e:
            return self._make_error(msg_id, -32603, str(e))

    @staticmethod
    def _make_response(msg_id: Any, result: Any) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": result,
        }

    @staticmethod
    def _make_error(msg_id: Any, code: int, message: str) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": code, "message": message},
        }

    @staticmethod
    def _send_response(pipe, response: dict[str, Any]) -> None:
        import win32file
        data = json.dumps(response).encode("utf-8")
        win32file.WriteFile(pipe, data)
