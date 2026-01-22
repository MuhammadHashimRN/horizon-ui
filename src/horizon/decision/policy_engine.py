"""Confidence gating and conflict resolution for fused actions."""

from __future__ import annotations

import logging
import time

from horizon.constants import DEFAULT_CONFIDENCE_THRESHOLD, GESTURE_COOLDOWN_MS
from horizon.event_bus import EventBus
from horizon.types import ActionType, Event, EventType, FusedAction, InputSource

logger = logging.getLogger(__name__)


class PolicyEngine:
    """Applies confidence gating, cooldowns, and conflict resolution.

    Subscribes to FUSED_ACTION events. Only passes actions that meet
    the confidence threshold and cooldown requirements. Publishes
    validated actions as OS_EVENT-ready FUSED_ACTION events.
    """

    def __init__(
        self,
        event_bus: EventBus,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        cooldown_ms: float = GESTURE_COOLDOWN_MS,
        voice_overrides_gesture: bool = True,
    ) -> None:
        self.event_bus = event_bus
        self.confidence_threshold = confidence_threshold
        self._cooldown_s = cooldown_ms / 1000.0
        self._voice_overrides_gesture = voice_overrides_gesture

        # Track last action time per action type for cooldown
        self._last_action_time: dict[ActionType, float] = {}
        # Track if voice recently overrode gesture
        self._voice_override_until: float = 0.0

        self.event_bus.subscribe(EventType.FUSED_ACTION, self._on_fused_action)
        logger.info("PolicyEngine initialized (threshold=%.2f)", confidence_threshold)

    def _on_fused_action(self, event: Event) -> None:
        if event.source == "policy_engine":
            return

        action: FusedAction = event.data
        now = time.time()

        # Confidence gate
        if action.confidence < self.confidence_threshold:
            logger.debug("Action %s rejected: low confidence %.2f", action.action.value, action.confidence)
            return

        # Voice override: suppress gesture actions during voice override window
        if (
            self._voice_overrides_gesture
            and action.source == InputSource.GESTURE
            and now < self._voice_override_until
        ):
            logger.debug("Gesture suppressed during voice override")
            return

        # Cooldown check (mouse_move is exempt)
        if action.action != ActionType.MOUSE_MOVE:
            last_time = self._last_action_time.get(action.action, 0.0)
            if now - last_time < self._cooldown_s:
                logger.debug("Action %s on cooldown", action.action.value)
                return

        # Record action time
        self._last_action_time[action.action] = now

        # Set voice override window
        if action.source == InputSource.VOICE:
            self._voice_override_until = now + 0.5  # 500ms override window

        # Pass validated action
        self.event_bus.publish(Event(
            type=EventType.OS_EVENT,
            data=action,
            source="policy_engine",
        ))

    def close(self) -> None:
        self.event_bus.unsubscribe(EventType.FUSED_ACTION, self._on_fused_action)
        logger.info("PolicyEngine closed")
