"""
Horizon UI Backend Test Suite
=============================
Tests each backend functionality individually with real hardware.
Run: python scripts/test_backend.py [test_name]
Available tests: all, gesture, voice, vad, asr, intent, fusion, eventbus, settings, plugins, metrics
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
import threading
from collections import deque

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import cv2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_backend")


# ─── Utilities ─────────────────────────────────────────────────────────────

def green(s): return f"\033[92m{s}\033[0m"
def red(s): return f"\033[91m{s}\033[0m"
def yellow(s): return f"\033[93m{s}\033[0m"
def cyan(s): return f"\033[96m{s}\033[0m"

def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {cyan(title)}")
    print(f"{'='*60}")

def print_result(name, passed, detail=""):
    status = green("PASS") if passed else red("FAIL")
    print(f"  [{status}] {name}" + (f" - {detail}" if detail else ""))
    return passed


# ─── Test: EventBus ────────────────────────────────────────────────────────

def test_eventbus():
    print_header("TEST: EventBus (pub/sub infrastructure)")
    from horizon.event_bus import EventBus
    from horizon.types import Event, EventType

    bus = EventBus()
    results = []
    received = []

    def handler(event):
        received.append(event)

    # Test subscribe + publish
    bus.subscribe(EventType.FRAME, handler)
    bus.publish(Event(type=EventType.FRAME, data="test_data"))
    results.append(print_result("Subscribe & publish", len(received) == 1, f"received {len(received)} events"))

    # Test unsubscribe
    bus.unsubscribe(EventType.FRAME, handler)
    bus.publish(Event(type=EventType.FRAME, data="should_not_receive"))
    results.append(print_result("Unsubscribe", len(received) == 1, "no event after unsub"))

    # Test multiple subscribers
    r1, r2 = [], []
    bus.subscribe(EventType.HAND_DETECTED, lambda e: r1.append(e))
    bus.subscribe(EventType.HAND_DETECTED, lambda e: r2.append(e))
    bus.publish(Event(type=EventType.HAND_DETECTED, data="multi"))
    results.append(print_result("Multiple subscribers", len(r1) == 1 and len(r2) == 1))

    # Test latency
    latencies = []
    def latency_handler(e):
        latencies.append(time.perf_counter() - e.data)
    bus.subscribe(EventType.GESTURE_RESULT, latency_handler)
    for _ in range(100):
        bus.publish(Event(type=EventType.GESTURE_RESULT, data=time.perf_counter()))
    avg_latency = sum(latencies) / len(latencies) * 1000
    results.append(print_result("Publish latency", avg_latency < 1.0, f"avg={avg_latency:.3f}ms"))

    bus.clear()
    return all(results)


# ─── Test: Gesture Recognition Pipeline ────────────────────────────────────

def test_gesture():
    print_header("TEST: Gesture Recognition Pipeline (camera required)")

    import mediapipe as mp
    from horizon.event_bus import EventBus
    from horizon.types import Event, EventType, GestureResult, GestureLabel

    bus = EventBus()
    results = []
    gestures_detected = []
    landmarks_received = []

    # Subscribe to events
    def on_gesture(event):
        if isinstance(event.data, GestureResult):
            gestures_detected.append(event.data)

    def on_landmarks(event):
        landmarks_received.append(event.data)

    bus.subscribe(EventType.GESTURE_RESULT, on_gesture)
    bus.subscribe(EventType.LANDMARKS, on_landmarks)

    # Initialize components
    from horizon.perception.hand_detector import HandDetector
    from horizon.perception.landmark_estimator import LandmarkEstimator
    from horizon.perception.gesture_classifier import GestureClassifier

    detector = HandDetector(event_bus=bus, max_num_hands=1)
    estimator = LandmarkEstimator(event_bus=bus)
    classifier = GestureClassifier(event_bus=bus, confidence_threshold=0.5)

    results.append(print_result("HandDetector init", True, "GestureRecognizer loaded"))
    results.append(print_result("LandmarkEstimator init", True))
    results.append(print_result("GestureClassifier init", True, "hybrid mode"))

    # Open camera and process frames
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(red("  ERROR: Cannot open camera. Skipping live gesture test."))
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"\n  {yellow('Show your hand to the camera for 5 seconds...')}")
    print(f"  {yellow('Try: open palm, fist, thumbs up, pointing up')}")

    frame_count = 0
    latencies = []
    start_time = time.time()
    duration = 5.0

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        bus.publish(Event(type=EventType.FRAME, data=frame))
        latencies.append((time.perf_counter() - t0) * 1000)
        frame_count += 1

        # Show live feedback
        gesture_text = "none"
        if gestures_detected:
            g = gestures_detected[-1]
            gesture_text = f"{g.label.value} ({g.confidence:.2f})"
        cv2.putText(frame, f"Gesture: {gesture_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Landmarks: {len(landmarks_received)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Gesture Test (press Q to quit early)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Report
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    results.append(print_result(
        "Frame processing",
        frame_count > 30,
        f"{frame_count} frames, avg latency={avg_latency:.1f}ms"
    ))
    results.append(print_result(
        "Landmarks detected",
        len(landmarks_received) > 0,
        f"{len(landmarks_received)} landmark sets"
    ))

    unique_gestures = set(g.label for g in gestures_detected if g.label != GestureLabel.NONE)
    results.append(print_result(
        "Gestures recognized",
        len(gestures_detected) > 0,
        f"{len(gestures_detected)} total, unique: {[g.value for g in unique_gestures]}"
    ))

    results.append(print_result(
        "Gesture latency < 120ms",
        avg_latency < 120,
        f"{avg_latency:.1f}ms (target: <120ms)"
    ))

    # Cleanup
    detector.close()
    estimator.close()
    classifier.close()
    bus.clear()

    return all(results)


# ─── Test: VAD (Voice Activity Detection) ──────────────────────────────────

def test_vad():
    print_header("TEST: Voice Activity Detection (microphone required)")

    from horizon.event_bus import EventBus
    from horizon.types import Event, EventType
    from horizon.perception.vad import VAD

    bus = EventBus()
    results = []
    speech_segments = []

    def on_speech(event):
        speech_segments.append(event.data)
        logger.info("Speech segment detected: %d bytes", len(event.data))

    bus.subscribe(EventType.SPEECH_SEGMENT, on_speech)

    vad = VAD(event_bus=bus, sample_rate=16000, aggressiveness=3)
    results.append(print_result("VAD init", True, "WebRTC VAD mode 3"))

    print(f"\n  {yellow('Speak into your microphone for 5 seconds...')}")
    print(f"  {yellow('Say something clearly, then pause.')}")

    import sounddevice as sd

    audio_chunks = []
    def audio_callback(indata, frames, time_info, status):
        pcm = (indata[:, 0] * 32767).astype(np.int16).tobytes()
        audio_chunks.append(pcm)
        bus.publish(Event(type=EventType.AUDIO_CHUNK, data=pcm))

    stream = sd.InputStream(
        samplerate=16000,
        channels=1,
        dtype='float32',
        blocksize=int(16000 * 0.03),
        callback=audio_callback,
    )

    with stream:
        time.sleep(5.0)

    results.append(print_result(
        "Audio capture",
        len(audio_chunks) > 100,
        f"{len(audio_chunks)} chunks captured"
    ))
    results.append(print_result(
        "Speech segments detected",
        len(speech_segments) > 0,
        f"{len(speech_segments)} segments"
    ))

    if speech_segments:
        total_bytes = sum(len(s) for s in speech_segments)
        duration_ms = total_bytes / (16000 * 2) * 1000
        results.append(print_result(
            "Speech segment quality",
            duration_ms > 250,
            f"total speech: {duration_ms:.0f}ms"
        ))

    vad.close()
    bus.clear()
    return all(results)


# ─── Test: ASR (Automatic Speech Recognition) ─────────────────────────────

def test_asr():
    print_header("TEST: Automatic Speech Recognition (microphone required)")

    from horizon.event_bus import EventBus
    from horizon.types import Event, EventType
    from horizon.perception.vad import VAD
    from horizon.perception.asr import ASR

    bus = EventBus()
    results = []
    transcripts = []

    def on_transcript(event):
        transcripts.append(event.data)
        logger.info("Transcript: %s", event.data)

    bus.subscribe(EventType.TRANSCRIPT, on_transcript)

    vad = VAD(event_bus=bus, sample_rate=16000)
    asr = ASR(event_bus=bus, backend="whisper", whisper_model="small", language="en")
    results.append(print_result("ASR init", True, "whisper-small loaded"))

    print(f"\n  {yellow('Say a clear command like:')}")
    print(f"  {yellow('  \"scroll down\" or \"open browser\" or \"click here\"')}")
    print(f"  {yellow('Recording for 7 seconds...')}")

    import sounddevice as sd

    def audio_callback(indata, frames, time_info, status):
        pcm = (indata[:, 0] * 32767).astype(np.int16).tobytes()
        bus.publish(Event(type=EventType.AUDIO_CHUNK, data=pcm))

    stream = sd.InputStream(
        samplerate=16000, channels=1, dtype='float32',
        blocksize=int(16000 * 0.03), callback=audio_callback,
    )

    with stream:
        time.sleep(7.0)

    # Wait a moment for ASR to finish processing
    time.sleep(2.0)

    results.append(print_result(
        "Transcription produced",
        len(transcripts) > 0,
        f"{len(transcripts)} transcripts: {transcripts[:3]}"
    ))

    if transcripts:
        results.append(print_result(
            "Transcript non-empty",
            any(len(t) > 0 for t in transcripts),
            f"sample: '{transcripts[0][:60]}'"
        ))

    vad.close()
    asr.close()
    bus.clear()
    return all(results)


# ─── Test: Intent Parser ──────────────────────────────────────────────────

def test_intent():
    print_header("TEST: Intent Parser (voice command matching)")

    from horizon.event_bus import EventBus
    from horizon.types import Event, EventType, ActionType
    from horizon.perception.intent_parser import IntentParser

    bus = EventBus()
    results = []
    intents = []

    def on_intent(event):
        intents.append(event.data)

    bus.subscribe(EventType.VOICE_INTENT, on_intent)

    parser = IntentParser(event_bus=bus)
    results.append(print_result("IntentParser init", True))

    # Test exact matches
    test_cases = [
        ("scroll down", ActionType.SCROLL_DOWN),
        ("scroll up", ActionType.SCROLL_UP),
        ("click", ActionType.LEFT_CLICK),
        ("right click", ActionType.RIGHT_CLICK),
        ("double click", ActionType.DOUBLE_CLICK),
        ("zoom in", ActionType.ZOOM_IN),
        ("zoom out", ActionType.ZOOM_OUT),
        ("volume up", ActionType.VOLUME_UP),
        ("volume down", ActionType.VOLUME_DOWN),
        ("take screenshot", ActionType.SCREENSHOT),
        ("minimize window", ActionType.MINIMIZE_WINDOW),
        ("maximize window", ActionType.MAXIMIZE_WINDOW),
    ]

    matched = 0
    for transcript, expected_action in test_cases:
        intents.clear()
        bus.publish(Event(type=EventType.TRANSCRIPT, data=transcript))
        if intents and intents[0].action == expected_action:
            matched += 1
            print_result(f"  '{transcript}'", True, f"-> {expected_action.value}")
        else:
            actual = intents[0].action.value if intents else "no match"
            print_result(f"  '{transcript}'", False, f"-> {actual} (expected {expected_action.value})")

    accuracy = matched / len(test_cases) * 100
    results.append(print_result(
        f"Intent accuracy",
        accuracy >= 80,
        f"{matched}/{len(test_cases)} ({accuracy:.0f}%)"
    ))

    # Test fuzzy matching
    intents.clear()
    bus.publish(Event(type=EventType.TRANSCRIPT, data="scrol down"))  # typo
    fuzzy_pass = len(intents) > 0
    results.append(print_result("Fuzzy matching", fuzzy_pass, "typo 'scrol down' matched" if fuzzy_pass else "no match"))

    parser.close()
    bus.clear()
    return all(results)


# ─── Test: Fusion Engine ──────────────────────────────────────────────────

def test_fusion():
    print_header("TEST: Multimodal Fusion Engine")

    from horizon.event_bus import EventBus
    from horizon.types import (
        Event, EventType, GestureResult, GestureLabel,
        VoiceIntent, ActionType, FusedAction, InputSource,
    )
    from horizon.decision.fusion_engine import FusionEngine

    bus = EventBus()
    results = []
    fused_actions = []

    def on_fused(event):
        fused_actions.append(event.data)

    bus.subscribe(EventType.FUSED_ACTION, on_fused)

    fusion = FusionEngine(event_bus=bus, alignment_window_ms=200)
    results.append(print_result("FusionEngine init", True))

    # Test gesture-only action (timestamp far in the past to avoid alignment)
    fused_actions.clear()
    t1 = time.time()
    bus.publish(Event(
        type=EventType.GESTURE_RESULT,
        data=GestureResult(label=GestureLabel.POINT, confidence=0.9, cursor_x=0.5, cursor_y=0.5, timestamp=t1),
    ))
    results.append(print_result(
        "Gesture-only -> MOUSE_MOVE",
        len(fused_actions) == 1 and fused_actions[0].action == ActionType.MOUSE_MOVE,
        f"action={fused_actions[0].action.value}" if fused_actions else "none"
    ))

    # Test voice-only action (timestamp 10s later so no alignment with previous gesture)
    fused_actions.clear()
    t2 = t1 + 10.0
    bus.publish(Event(
        type=EventType.VOICE_INTENT,
        data=VoiceIntent(action=ActionType.SCROLL_DOWN, confidence=0.95, transcript="scroll down", timestamp=t2),
    ))
    results.append(print_result(
        "Voice-only -> SCROLL_DOWN",
        len(fused_actions) == 1 and fused_actions[0].action == ActionType.SCROLL_DOWN,
    ))

    # Test voice + gesture fusion (voice should win)
    # Publish voice FIRST so it's in _recent_voice when gesture arrives
    fused_actions.clear()
    t3 = t2 + 10.0
    bus.publish(Event(
        type=EventType.VOICE_INTENT,
        data=VoiceIntent(action=ActionType.ZOOM_IN, confidence=0.9, transcript="zoom in", timestamp=t3),
    ))
    bus.publish(Event(
        type=EventType.GESTURE_RESULT,
        data=GestureResult(label=GestureLabel.FIST, confidence=0.85, timestamp=t3),
    ))
    # _on_gesture finds aligned voice -> publishes ZOOM_IN (voice wins)
    voice_won = any(a.action == ActionType.ZOOM_IN for a in fused_actions)
    results.append(print_result("Multimodal fusion (voice wins)", voice_won))

    # Test NONE gesture ignored
    fused_actions.clear()
    bus.publish(Event(
        type=EventType.GESTURE_RESULT,
        data=GestureResult(label=GestureLabel.NONE, confidence=0.3),
    ))
    results.append(print_result("NONE gesture ignored", len(fused_actions) == 0))

    fusion.close()
    bus.clear()
    return all(results)


# ─── Test: Settings & Profiles ─────────────────────────────────────────────

def test_settings():
    print_header("TEST: Settings Store & Profile Manager")

    from horizon.profiles.settings_store import SettingsStore
    from horizon.profiles.profile_manager import ProfileManager
    from horizon.event_bus import EventBus

    results = []

    store = SettingsStore()
    results.append(print_result("SettingsStore init", True))

    # Test reading defaults
    camera = store.get("camera")
    results.append(print_result("Read camera settings", camera is not None, f"device_index={camera.get('device_index')}"))

    fps = store.get("camera", "fps", 30)
    results.append(print_result("Read nested setting", fps == 30, f"fps={fps}"))

    # Test setting values
    store.set("camera", "fps", 60)
    results.append(print_result("Set value", store.get("camera", "fps") == 60))

    # Test profile manager
    bus = EventBus()
    pm = ProfileManager(event_bus=bus, settings_store=store)
    results.append(print_result("ProfileManager init", True, f"{len(pm._profiles)} profiles loaded"))

    profiles_loaded = list(pm._profiles.keys())
    results.append(print_result("Profiles discovered", len(profiles_loaded) >= 3,
                                f"profiles: {profiles_loaded}"))

    # Test profile activation
    if "sterile" in pm._profiles:
        pm.activate("sterile")
        sterile_threshold = store.get("gesture", "confidence_threshold")
        results.append(print_result("Profile 'sterile' activated",
                                    sterile_threshold is not None,
                                    f"confidence_threshold={sterile_threshold}"))
        pm.deactivate()

    bus.clear()
    return all(results)


# ─── Test: Plugins ─────────────────────────────────────────────────────────

def test_plugins():
    print_header("TEST: Plugin System")

    from horizon.event_bus import EventBus
    from horizon.control.plugin_manager import PluginManager

    bus = EventBus()
    results = []

    pm = PluginManager(event_bus=bus)
    results.append(print_result("PluginManager init", True))

    # Discover plugins
    discovered = pm.discover()
    results.append(print_result("Plugin discovery", len(discovered) >= 2,
                                f"found: {discovered}"))

    # Load plugins
    for name in discovered:
        loaded = pm.load(name)
        results.append(print_result(f"Load plugin '{name}'", loaded))

    # Check manifests
    for name, plugin in pm._plugins.items():
        manifest = plugin.get_manifest()
        results.append(print_result(
            f"Plugin '{name}' manifest",
            "name" in manifest and "version" in manifest,
            f"v{manifest.get('version', '?')}"
        ))

    bus.clear()
    return all(results)


# ─── Test: Performance Metrics ─────────────────────────────────────────────

def test_metrics():
    print_header("TEST: Performance Metrics Collection")

    import psutil
    from horizon.event_bus import EventBus
    from horizon.types import Event, EventType, GestureResult, GestureLabel

    bus = EventBus()
    results = []
    process = psutil.Process(os.getpid())

    # Measure baseline
    cpu_before = process.cpu_percent(interval=0.5)
    mem_before = process.memory_info().rss / (1024 * 1024)
    print(f"  Baseline: CPU={cpu_before:.1f}%, RAM={mem_before:.1f}MB")

    # Initialize perception components
    from horizon.perception.hand_detector import HandDetector
    from horizon.perception.landmark_estimator import LandmarkEstimator
    from horizon.perception.gesture_classifier import GestureClassifier

    detector = HandDetector(event_bus=bus)
    estimator = LandmarkEstimator(event_bus=bus)
    classifier = GestureClassifier(event_bus=bus)

    mem_after_load = process.memory_info().rss / (1024 * 1024)
    print(f"  After model load: RAM={mem_after_load:.1f}MB (+{mem_after_load - mem_before:.1f}MB)")
    results.append(print_result("RAM < 3.5GB", mem_after_load < 3500, f"{mem_after_load:.0f}MB"))

    # Process frames and measure latency
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(red("  Camera not available for metrics test"))
        detector.close()
        estimator.close()
        classifier.close()
        bus.clear()
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"\n  {yellow('Measuring latency over 3 seconds (show hand)...')}")
    frame_latencies = []
    start = time.time()

    while time.time() - start < 3.0:
        ret, frame = cap.read()
        if not ret:
            break
        t0 = time.perf_counter()
        bus.publish(Event(type=EventType.FRAME, data=frame))
        frame_latencies.append((time.perf_counter() - t0) * 1000)

    cap.release()

    if frame_latencies:
        avg_lat = sum(frame_latencies) / len(frame_latencies)
        p95_lat = sorted(frame_latencies)[int(len(frame_latencies) * 0.95)]
        max_lat = max(frame_latencies)
        fps = len(frame_latencies) / 3.0

        print(f"\n  --- Gesture Pipeline Latency ---")
        print(f"  Frames:     {len(frame_latencies)}")
        print(f"  FPS:        {fps:.1f}")
        print(f"  Avg:        {avg_lat:.1f}ms")
        print(f"  P95:        {p95_lat:.1f}ms")
        print(f"  Max:        {max_lat:.1f}ms")

        results.append(print_result("Avg latency < 120ms", avg_lat < 120, f"{avg_lat:.1f}ms"))
        results.append(print_result("P95 latency < 200ms", p95_lat < 200, f"{p95_lat:.1f}ms"))
        results.append(print_result("FPS >= 15", fps >= 15, f"{fps:.1f} FPS"))

    cpu_during = process.cpu_percent(interval=1.0)
    results.append(print_result("CPU < 50%", cpu_during < 50, f"{cpu_during:.1f}%"))

    mem_final = process.memory_info().rss / (1024 * 1024)
    results.append(print_result("Final RAM check", mem_final < 3500, f"{mem_final:.0f}MB"))

    detector.close()
    estimator.close()
    classifier.close()
    bus.clear()

    return all(results)


# ─── Main ──────────────────────────────────────────────────────────────────

TESTS = {
    "eventbus": test_eventbus,
    "gesture": test_gesture,
    "vad": test_vad,
    "asr": test_asr,
    "intent": test_intent,
    "fusion": test_fusion,
    "settings": test_settings,
    "plugins": test_plugins,
    "metrics": test_metrics,
}

def main():
    parser = argparse.ArgumentParser(description="Horizon UI Backend Test Suite")
    parser.add_argument("test", nargs="?", default="all",
                        choices=["all"] + list(TESTS.keys()),
                        help="Which test to run (default: all)")
    args = parser.parse_args()

    if args.test == "all":
        tests_to_run = list(TESTS.items())
    else:
        tests_to_run = [(args.test, TESTS[args.test])]

    total_pass = 0
    total_fail = 0

    for name, test_fn in tests_to_run:
        try:
            passed = test_fn()
            if passed:
                total_pass += 1
            else:
                total_fail += 1
        except Exception as e:
            print(red(f"\n  EXCEPTION in {name}: {e}"))
            import traceback
            traceback.print_exc()
            total_fail += 1

    print(f"\n{'='*60}")
    print(f"  RESULTS: {green(f'{total_pass} passed')}, {red(f'{total_fail} failed') if total_fail else green('0 failed')}")
    print(f"{'='*60}\n")

    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
