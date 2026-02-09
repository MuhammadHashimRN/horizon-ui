# Horizon UI Architecture

## 5-Layer Architecture

```
┌─────────────────────────────────────────────┐
│           Presentation Layer                │
│  Overlay UI │ HUD │ Settings │ System Tray  │
├─────────────────────────────────────────────┤
│             Control Layer                   │
│  OS Event Injector │ Plugin Manager │ IPC   │
├─────────────────────────────────────────────┤
│             Decision Layer                  │
│  Fusion Engine │ Policy Engine │ Context    │
├─────────────────────────────────────────────┤
│            Perception Layer                 │
│  Hand Detector │ Gesture Classifier │ ASR   │
├─────────────────────────────────────────────┤
│             Capture Layer                   │
│       Frame Capture │ Audio Capture         │
└─────────────────────────────────────────────┘
```

## Data Flow

```
Camera → FrameCapture → HandDetector → LandmarkEstimator
    → FeatureExtractor → GestureClassifier → FusionEngine
    → PolicyEngine → OSEventInjector → Win32 SendInput

Microphone → AudioCapture → VAD → ASR → IntentParser
    → FusionEngine → PolicyEngine → OSEventInjector
```

## Event Bus

All layers communicate via a central EventBus (pub/sub pattern):

- `FRAME` — Camera frame captured
- `AUDIO_CHUNK` — Audio chunk captured
- `HAND_DETECTED` — Hand detection result
- `LANDMARKS` — Normalized landmark set
- `GESTURE_RESULT` — Classified gesture
- `SPEECH_SEGMENT` — Speech audio segment
- `TRANSCRIPT` — ASR transcript
- `VOICE_INTENT` — Parsed voice intent
- `FUSED_ACTION` — Combined gesture+voice action
- `OS_EVENT` — Validated action for OS injection

## Threading Model

- **Main thread**: PyQt6 event loop + overlay rendering
- **FrameCapture thread**: Camera read loop
- **Audio thread**: sounddevice callback (separate thread managed by PortAudio)
- **IPC thread**: Named Pipe server for plugin communication

Perception, decision, and control run on the publisher's thread
(synchronous EventBus callbacks) to minimize latency.
