"""Application orchestrator — initializes and coordinates all layers."""

from __future__ import annotations

import logging
import signal
import sys
import threading

from horizon.event_bus import EventBus
from horizon.profiles.settings_store import SettingsStore
from horizon.types import Event, EventType

logger = logging.getLogger(__name__)


class App:
    """Master orchestrator for Horizon UI.

    Initializes all layers in order:
    EventBus → Config → Capture → Perception → Decision → Control → Presentation

    Manages thread lifecycle and graceful shutdown.
    """

    def __init__(
        self,
        config_path: str | None = None,
        profile: str | None = None,
        log_level: str = "INFO",
        no_overlay: bool = False,
    ) -> None:
        self._no_overlay = no_overlay
        self._is_active = True
        self._components: list = []

        # Setup logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper(), logging.INFO),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )

        # Core infrastructure
        self.event_bus = EventBus()
        self.settings = SettingsStore(defaults_file=config_path)

        # Apply profile if specified
        self._profile_name = profile

        logger.info("Horizon UI initializing...")

    def _init_capture_layer(self) -> None:
        from horizon.capture.audio_capture import AudioCapture
        from horizon.capture.frame_capture import FrameCapture

        cam = self.settings.get("camera")
        self._frame_capture = FrameCapture(
            event_bus=self.event_bus,
            device_index=cam.get("device_index", 0),
            resolution=tuple(cam.get("resolution", [640, 480])),
            fps=cam.get("fps", 30),
        )

        audio = self.settings.get("audio")
        self._audio_capture = AudioCapture(
            event_bus=self.event_bus,
            sample_rate=audio.get("sample_rate", 16000),
            channels=audio.get("channels", 1),
            chunk_duration_ms=audio.get("chunk_duration_ms", 30),
        )

        self._components.extend([self._frame_capture, self._audio_capture])
        logger.info("Capture layer initialized")

    def _init_perception_layer(self) -> None:
        from horizon.perception.asr import ASR
        from horizon.perception.coordinate_mapper import CoordinateMapper
        from horizon.perception.cursor_tracker import CursorTracker
        from horizon.perception.feature_extractor import FeatureExtractor
        from horizon.perception.gesture_classifier import GestureClassifier
        from horizon.perception.hand_detector import HandDetector
        from horizon.perception.intent_parser import IntentParser
        from horizon.perception.landmark_estimator import LandmarkEstimator
        from horizon.perception.vad import VAD

        mp_cfg = self.settings.get("mediapipe")
        gesture_cfg = self.settings.get("gesture")
        asr_cfg = self.settings.get("asr")

        self._hand_detector = HandDetector(
            event_bus=self.event_bus,
            max_num_hands=mp_cfg.get("max_num_hands", 1),
            min_detection_confidence=mp_cfg.get("min_detection_confidence", 0.7),
            min_tracking_confidence=mp_cfg.get("min_tracking_confidence", 0.6),
        )

        self._landmark_estimator = LandmarkEstimator(event_bus=self.event_bus)

        self._feature_extractor = FeatureExtractor(
            event_bus=self.event_bus,
            temporal_window=gesture_cfg.get("temporal_window_frames", 15),
        )

        # CoordinateMapper with adaptive smoothing + webcam mirroring
        self._coordinate_mapper = CoordinateMapper(
            smoothing_alpha=gesture_cfg.get("cursor_smoothing_alpha", 0.5),
            mirror_x=gesture_cfg.get("mirror_x", False),
            adaptive_smoothing=True,
        )

        # CursorTracker: moves cursor on every frame (decoupled from gestures)
        self._cursor_tracker = CursorTracker(
            event_bus=self.event_bus,
            coordinate_mapper=self._coordinate_mapper,
        )

        self._gesture_classifier = GestureClassifier(
            event_bus=self.event_bus,
            confidence_threshold=gesture_cfg.get("confidence_threshold", 0.75),
        )

        self._vad = VAD(
            event_bus=self.event_bus,
            sample_rate=self.settings.get("audio", "sample_rate", 16000),
        )

        self._asr = ASR(
            event_bus=self.event_bus,
            backend=asr_cfg.get("backend", "whisper"),
            whisper_model=asr_cfg.get("whisper_model", "small"),
            language=asr_cfg.get("language", "en"),
        )

        self._intent_parser = IntentParser(event_bus=self.event_bus)

        self._components.extend([
            self._hand_detector, self._landmark_estimator,
            self._feature_extractor, self._cursor_tracker,
            self._gesture_classifier,
            self._vad, self._asr, self._intent_parser,
        ])
        logger.info("Perception layer initialized")

    def _init_decision_layer(self) -> None:
        from horizon.decision.action_mapper import ActionMapper
        from horizon.decision.context_manager import ContextManager
        from horizon.decision.fusion_engine import FusionEngine
        from horizon.decision.policy_engine import PolicyEngine

        gesture_cfg = self.settings.get("gesture")

        self._fusion_engine = FusionEngine(event_bus=self.event_bus)
        self._policy_engine = PolicyEngine(
            event_bus=self.event_bus,
            confidence_threshold=gesture_cfg.get("confidence_threshold", 0.75),
        )
        self._action_mapper = ActionMapper()
        self._context_manager = ContextManager()

        self._components.extend([
            self._fusion_engine, self._policy_engine,
        ])
        logger.info("Decision layer initialized")

    def _init_control_layer(self) -> None:
        from horizon.control.ipc_auth import IPCAuth
        from horizon.control.ipc_server import IPCServer
        from horizon.control.os_event_injector import OSEventInjector
        from horizon.control.plugin_manager import PluginManager

        self._os_injector = OSEventInjector(event_bus=self.event_bus)
        self._plugin_manager = PluginManager(event_bus=self.event_bus)

        self._ipc_auth = IPCAuth()
        self._ipc_server = IPCServer(auth=self._ipc_auth)

        self._components.extend([self._os_injector, self._plugin_manager])
        logger.info("Control layer initialized")

    def _init_presentation_layer(self) -> None:
        if self._no_overlay:
            logger.info("Overlay disabled by --no-overlay flag")
            return

        from horizon.presentation.audio_feedback import AudioFeedback
        from horizon.presentation.overlay_ui import OverlayWindow
        from horizon.presentation.tray_icon import TrayIcon

        overlay_cfg = self.settings.get("overlay")

        self._overlay = OverlayWindow(
            event_bus=self.event_bus,
            theme_name=overlay_cfg.get("theme", "dark"),
            opacity=overlay_cfg.get("opacity", 0.85),
            show_skeleton=overlay_cfg.get("show_hand_skeleton", True),
            cursor_size=overlay_cfg.get("cursor_size", 24),
        )

        self._audio_feedback = AudioFeedback()
        logger.info("Presentation layer initialized")

    def _init_profiles(self) -> None:
        from horizon.profiles.profile_manager import ProfileManager

        self._profile_manager = ProfileManager(
            event_bus=self.event_bus,
            settings_store=self.settings,
        )

        if self._profile_name:
            self._profile_manager.activate(self._profile_name)

        logger.info("Profiles initialized")

    def _setup_debug_listeners(self) -> None:
        """Subscribe to key events for real-time console visibility."""
        import sys
        from horizon.types import GestureResult, GestureLabel, FusedAction, VoiceIntent

        self._debug_counters = {"frames": 0, "hands": 0, "gestures": 0, "vad": 0, "transcripts": 0}

        def _print(msg: str) -> None:
            print(msg, flush=True)

        def on_frame(event: Event) -> None:
            self._debug_counters["frames"] += 1
            n = self._debug_counters["frames"]
            if n == 1:
                _print("  [STATUS] First frame captured from camera")
            elif n % 300 == 0:  # every ~10 seconds at 30fps
                c = self._debug_counters
                _print(f"  [STATUS] frames={c['frames']} hands={c['hands']} gestures={c['gestures']} vad_segments={c['vad']} transcripts={c['transcripts']}")

        def on_gesture(event: Event) -> None:
            data = event.data
            if isinstance(data, GestureResult) and data.label != GestureLabel.NONE:
                self._debug_counters["gestures"] += 1
                _print(f"  [GESTURE] {data.label.value} (conf={data.confidence:.2f}) cursor=({data.cursor_x:.3f}, {data.cursor_y:.3f})")

        def on_voice_intent(event: Event) -> None:
            data = event.data
            if isinstance(data, VoiceIntent):
                _print(f"  [VOICE] action={data.action.value} conf={data.confidence:.2f} params={data.params}")

        def on_transcript(event: Event) -> None:
            text = event.data if isinstance(event.data, str) else str(event.data)
            if text.strip():
                _print(f"  [TRANSCRIPT] \"{text.strip()}\"")

        def on_fused(event: Event) -> None:
            data = event.data
            if isinstance(data, FusedAction):
                _print(f"  [ACTION] {data.action.value} source={data.source.value} conf={data.confidence:.2f}")

        def on_os_event(event: Event) -> None:
            data = event.data
            if isinstance(data, FusedAction):
                params_str = f" params={data.params}" if data.params else ""
                _print(f"  >>> [EXECUTED] {data.action.value}{params_str}")

        def on_hand(event: Event) -> None:
            data = event.data
            if data.get("has_hands"):
                self._debug_counters["hands"] += 1
                labels = data.get("gesture_labels", [])
                if labels:
                    top = labels[0]
                    if top["label"] != "None":
                        _print(f"  [HAND] detected: {top['label']} (score={top['score']:.2f})")

        def on_speech_segment(event: Event) -> None:
            self._debug_counters["vad"] += 1
            _print("  [VAD] Speech segment detected - sending to ASR...")

        def on_transcript_count(event: Event) -> None:
            text = event.data if isinstance(event.data, str) else str(event.data)
            if text.strip():
                self._debug_counters["transcripts"] += 1

        self.event_bus.subscribe(EventType.FRAME, on_frame)
        self.event_bus.subscribe(EventType.GESTURE_RESULT, on_gesture)
        self.event_bus.subscribe(EventType.VOICE_INTENT, on_voice_intent)
        self.event_bus.subscribe(EventType.TRANSCRIPT, on_transcript)
        self.event_bus.subscribe(EventType.TRANSCRIPT, on_transcript_count)
        self.event_bus.subscribe(EventType.FUSED_ACTION, on_fused)
        self.event_bus.subscribe(EventType.OS_EVENT, on_os_event)
        self.event_bus.subscribe(EventType.HAND_DETECTED, on_hand)
        self.event_bus.subscribe(EventType.SPEECH_SEGMENT, on_speech_segment)
        logger.info("Debug event listeners active - events will appear in console")

    def start(self) -> None:
        """Initialize all layers and start processing."""
        self._init_capture_layer()
        self._init_perception_layer()
        self._init_decision_layer()
        self._init_control_layer()
        self._init_profiles()
        self._init_presentation_layer()
        self._setup_debug_listeners()

        # Start capture threads
        self._frame_capture.start()
        self._audio_capture.start()

        # Start IPC server
        self._ipc_server.start()

        # Discover and load built-in plugins
        for plugin_name in self._plugin_manager.discover():
            self._plugin_manager.load(plugin_name)

        logger.info("Horizon UI started - all layers active")

        # Show overlay
        if not self._no_overlay and hasattr(self, "_overlay"):
            self._overlay.show()

        self._audio_feedback.play_activate() if hasattr(self, "_audio_feedback") else None

    def toggle(self) -> None:
        """Toggle active/paused state."""
        self._is_active = not self._is_active
        state = "active" if self._is_active else "paused"
        logger.info("Horizon UI %s", state)

        self.event_bus.publish(Event(
            type=EventType.SYSTEM_STATE,
            data={"active": self._is_active},
            source="app",
        ))

    def shutdown(self) -> None:
        """Gracefully shut down all components."""
        logger.info("Shutting down Horizon UI...")

        if hasattr(self, "_audio_feedback"):
            self._audio_feedback.play_deactivate()

        # Stop capture
        if hasattr(self, "_frame_capture"):
            self._frame_capture.stop()
        if hasattr(self, "_audio_capture"):
            self._audio_capture.stop()

        # Stop IPC
        if hasattr(self, "_ipc_server"):
            self._ipc_server.stop()

        # Close components with close() method
        for component in reversed(self._components):
            if hasattr(component, "close"):
                try:
                    component.close()
                except Exception:
                    logger.exception("Error closing %s", type(component).__name__)

        # Close overlay
        if not self._no_overlay and hasattr(self, "_overlay"):
            self._overlay.close_overlay()

        # Clear event bus
        self.event_bus.clear()

        logger.info("Horizon UI shutdown complete")
