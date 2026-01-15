"""Rule-based intent parser for voice command transcripts."""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import yaml

from horizon.constants import CONFIG_DIR, VOICE_COMMANDS_FILE
from horizon.event_bus import EventBus
from horizon.types import ActionType, Event, EventType, VoiceIntent

logger = logging.getLogger(__name__)

# Minimum fuzzy match ratio to accept a command
MIN_MATCH_RATIO = 0.75


class IntentParser:
    """Parses voice transcripts into structured VoiceIntent actions.

    Uses pattern matching and fuzzy matching against the voice commands
    configuration file. Subscribes to TRANSCRIPT events and publishes
    VOICE_INTENT events.
    """

    def __init__(
        self,
        event_bus: EventBus,
        commands_file: str | Path | None = None,
    ) -> None:
        self.event_bus = event_bus
        self._commands: list[dict[str, Any]] = []

        path = Path(commands_file) if commands_file else Path(CONFIG_DIR) / VOICE_COMMANDS_FILE
        self._load_commands(path)

        self.event_bus.subscribe(EventType.TRANSCRIPT, self._on_transcript)
        logger.info("IntentParser initialized (%d commands)", len(self._commands))

    def _load_commands(self, path: Path) -> None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            self._commands = data.get("commands", [])
        except FileNotFoundError:
            logger.warning("Voice commands file not found: %s", path)
        except Exception:
            logger.exception("Failed to load voice commands from %s", path)

    def _on_transcript(self, event: Event) -> None:
        transcript: str = event.data
        intent = self.parse(transcript)
        if intent:
            self.event_bus.publish(Event(
                type=EventType.VOICE_INTENT,
                data=intent,
                source="intent_parser",
            ))

    def parse(self, transcript: str) -> VoiceIntent | None:
        transcript = transcript.strip().lower()
        if not transcript:
            return None

        best_match: VoiceIntent | None = None
        best_score = 0.0

        for cmd in self._commands:
            patterns = cmd.get("pattern", [])
            action_str = cmd.get("action", "")
            param_names = cmd.get("params", [])

            for pattern in patterns:
                score, extracted_params = self._match_pattern(transcript, pattern, param_names)
                if score > best_score and score >= MIN_MATCH_RATIO:
                    best_score = score

                    try:
                        action = ActionType(action_str)
                    except ValueError:
                        logger.warning("Unknown action type: %s", action_str)
                        continue

                    params = dict(extracted_params)
                    # Add static params from command definition
                    if "keys" in cmd:
                        params["keys"] = cmd["keys"]
                    if "key" in cmd:
                        params["key"] = cmd["key"]
                    if "profile" in cmd:
                        params["profile"] = cmd["profile"]

                    best_match = VoiceIntent(
                        action=action,
                        transcript=transcript,
                        confidence=score,
                        params=params,
                    )

        if best_match:
            logger.debug("Parsed intent: %s (%.2f)", best_match.action.value, best_match.confidence)

        return best_match

    def _match_pattern(
        self,
        transcript: str,
        pattern: str,
        param_names: list[str],
    ) -> tuple[float, list[tuple[str, str]]]:
        """Match transcript against a pattern, returning score and extracted params."""
        pattern_lower = pattern.lower()
        extracted: list[tuple[str, str]] = []

        # Check for parameterized patterns like "open {app}"
        param_regex = re.sub(r"\{(\w+)\}", r"(.+)", pattern_lower)

        match = re.fullmatch(param_regex, transcript)
        if match:
            for i, name in enumerate(param_names):
                if i < len(match.groups()):
                    extracted.append((name, match.group(i + 1).strip()))
            return 1.0, extracted

        # Try partial match
        match = re.search(param_regex, transcript)
        if match:
            for i, name in enumerate(param_names):
                if i < len(match.groups()):
                    extracted.append((name, match.group(i + 1).strip()))
            return 0.9, extracted

        # Fuzzy match (no parameter extraction)
        # Remove parameter placeholders for fuzzy comparison
        clean_pattern = re.sub(r"\{(\w+)\}", "", pattern_lower).strip()
        ratio = SequenceMatcher(None, transcript, clean_pattern).ratio()
        return ratio, []

    def close(self) -> None:
        self.event_bus.unsubscribe(EventType.TRANSCRIPT, self._on_transcript)
        logger.info("IntentParser closed")
