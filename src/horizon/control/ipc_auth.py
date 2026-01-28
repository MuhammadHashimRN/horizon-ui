"""Token-based authentication for plugin IPC connections."""

from __future__ import annotations

import hashlib
import logging
import secrets
from typing import Any

logger = logging.getLogger(__name__)


class IPCAuth:
    """Manages authentication tokens for plugin IPC connections.

    Generates secure tokens for plugins and validates incoming
    connection tokens before allowing access.
    """

    def __init__(self) -> None:
        self._tokens: dict[str, str] = {}  # plugin_name -> token
        self._master_secret = secrets.token_hex(32)
        logger.info("IPCAuth initialized")

    def generate_token(self, plugin_name: str) -> str:
        """Generate a unique authentication token for a plugin."""
        raw = f"{self._master_secret}:{plugin_name}:{secrets.token_hex(16)}"
        token = hashlib.sha256(raw.encode()).hexdigest()
        self._tokens[plugin_name] = token
        logger.debug("Token generated for plugin: %s", plugin_name)
        return token

    def validate_token(self, token: str) -> str | None:
        """Validate a token. Returns plugin name if valid, None otherwise."""
        for plugin_name, stored_token in self._tokens.items():
            if secrets.compare_digest(token, stored_token):
                return plugin_name
        return None

    def revoke_token(self, plugin_name: str) -> None:
        """Revoke a plugin's authentication token."""
        self._tokens.pop(plugin_name, None)
        logger.debug("Token revoked for plugin: %s", plugin_name)

    def revoke_all(self) -> None:
        self._tokens.clear()
        logger.info("All tokens revoked")
