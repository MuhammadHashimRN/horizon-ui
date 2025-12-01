# Horizon UI

**A software-only multimodal desktop overlay that enables touchless computer control through real-time hand gesture recognition and on-device speech recognition.**

Horizon UI runs as a transparent overlay on top of your desktop, capturing webcam input to track hand movements and microphone input to recognize voice commands. It translates these inputs into native OS events (mouse movement, clicks, keyboard input, scrolling, zooming) — allowing users to operate their computer without physically touching any input device.

---

## Table of Contents

- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Supported Gestures](#supported-gestures)
- [Supported Voice Commands](#supported-voice-commands)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Accessibility Profiles](#accessibility-profiles)
- [Plugin System](#plugin-system)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [ML Training Pipeline](#ml-training-pipeline)
- [Testing](#testing)
- [Performance Targets](#performance-targets)
- [Contributing](#contributing)
- [License](#license)

---

## Key Features

### Gesture Control
- **21-point hand landmark detection** using MediaPipe Hands, tracking each finger joint and wrist in real time
- **10 distinct gestures** recognized via a GRU/1D-CNN temporal classifier running on ONNX Runtime
- **Cursor control** by pointing your index finger — the virtual cursor follows your hand position on screen
- **Click, scroll, zoom, swipe, and drag** operations mapped to natural hand gestures
- **EMA + Kalman smoothing** filters to eliminate jitter and produce fluid cursor movement
- **Camera-to-screen calibration** wizard using 4-point homography for accurate coordinate mapping

### Voice Control
- **On-device speech recognition** — all audio processing happens locally, no cloud APIs required
- **Dual ASR backend** — choose between faster-whisper (CTranslate2) or Vosk for speech-to-text
- **WebRTC Voice Activity Detection** to filter silence and reduce unnecessary ASR processing
- **30+ built-in voice commands** covering application control, navigation, text input, media, and system actions
- **Rule-based intent parsing** with fuzzy matching for natural language flexibility
- **Custom command support** via YAML configuration

### Multimodal Fusion
- **Temporal alignment** — gesture and voice signals occurring within a 100ms window are fused together
- **Confidence-weighted decision making** — actions only execute when confidence exceeds configurable thresholds
- **Voice-overrides-gesture** policy — voice commands take priority during simultaneous input to prevent conflicts
- **Cooldown timers** — prevent rapid-fire repeated actions from accidental gestures

### Transparent Overlay HUD
- **Full-screen transparent overlay** built with PyQt6 — stays on top of all windows
- **Complete mouse passthrough** — the overlay never intercepts your clicks (uses Win32 `WS_EX_TRANSPARENT`)
- **Real-time HUD rendering** showing virtual cursor, hand skeleton, gesture label, confidence bar, voice transcript, and system status
- **Three themes** — Dark, Light, and High Contrast (WCAG 2.1 AA compliant)
- **System tray icon** with quick-access controls (pause/resume, settings, calibrate, quit)
- **Audio feedback** — sound cues for activation, deactivation, and error states

### Plugin System
- **JSON-RPC 2.0 protocol** over Windows Named Pipes for inter-process communication
- **Token-based authentication** — plugins must authenticate before accessing system events
- **Crash isolation** — plugins run in separate processes; a plugin crash cannot bring down the main application
- **Built-in plugins** for media playback control and presentation slide navigation

### Accessibility
- **Sterile Mode** — optimized for touchless operation in sterile environments (e.g., surgical settings), with higher confidence thresholds and audio feedback
- **Low Vision Mode** — enlarged cursor (48px), high-contrast theme, full opacity, verbose audio cues (WCAG 2.1 AA)
- **Voice-Only Mode** — disables camera and gesture tracking entirely, operating purely on voice commands
- **Hot-switchable profiles** — switch between accessibility profiles at runtime via voice command or settings

---

## System Architecture

Horizon UI follows a strict **5-layer architecture** where each layer communicates through a central publish/subscribe **Event Bus**. This decouples all components, making each layer independently testable and replaceable.

```
+-----------------------------------------------------------------+
|                     PRESENTATION LAYER                          |
|  OverlayUI  |  HUD Renderer  |  Settings Panel  |  System Tray |
+-----------------------------------------------------------------+
|                       CONTROL LAYER                             |
|   OS Event Injector  |  Plugin Manager  |  IPC Server + Auth    |
+-----------------------------------------------------------------+
|                       DECISION LAYER                            |
|   Fusion Engine  |  Policy Engine  |  Action Mapper  |  Context |
+-----------------------------------------------------------------+
|                      PERCEPTION LAYER                           |
|  Hand Detector | Landmarks | Features | Gesture Classifier      |
|  VAD  |  ASR (Whisper/Vosk)  |  Intent Parser                  |
+-----------------------------------------------------------------+
|                       CAPTURE LAYER                             |
|          Frame Capture (OpenCV)  |  Audio Capture (sounddevice) |
+-----------------------------------------------------------------+
                              ^
                    +---------+----------+
                    |     EVENT BUS      |
                    |  (Pub/Sub System)  |
                    +--------------------+
```

### Data Flow

**Gesture Pipeline:**
```
Webcam -> FrameCapture (30fps, 640x480)
  -> HandDetector (MediaPipe, 21 landmarks)
  -> LandmarkEstimator (normalize to LandmarkSet)
  -> FeatureExtractor (15-frame temporal buffer, velocity, angles, distances)
  -> GestureClassifier (ONNX Runtime, GRU/1D-CNN)
  -> FusionEngine -> PolicyEngine -> OSEventInjector -> Win32 SendInput
```

**Voice Pipeline:**
```
Microphone -> AudioCapture (16kHz, mono, int16 PCM)
  -> VAD (WebRTC, 30ms frames, speech onset/offset detection)
  -> ASR (faster-whisper or Vosk, on-device transcription)
  -> IntentParser (rule-based + fuzzy matching against command vocabulary)
  -> FusionEngine -> PolicyEngine -> OSEventInjector -> Win32 SendInput
```

### Threading Model

| Thread | Responsibility |
|---|---|
| **Main thread** | PyQt6 event loop, overlay rendering at ~30 FPS |
| **FrameCapture thread** | Camera read loop, bounded queue (maxsize=2, drops oldest) |
| **Audio thread** | sounddevice PortAudio callback, 30ms chunk buffering |
| **IPC thread** | Named Pipe server for plugin connections |
| **Perception/Decision/Control** | Synchronous callbacks on publisher's thread (minimizes latency) |

---

## Supported Gestures

| Gesture | Action | Description |
|---|---|---|
| **Point** (index finger) | Cursor movement | Index finger tip position maps to screen cursor |
| **Pinch** (thumb + index) | Left click | Thumb-index pinch triggers a left mouse click |
| **Fist** (closed hand) | Right click | Closed fist triggers a right mouse click |
| **Open Palm** | Pause tracking | Open palm pauses all gesture tracking |
| **Swipe Left** | Browser back | Horizontal swipe left triggers Alt+Left |
| **Swipe Right** | Browser forward | Horizontal swipe right triggers Alt+Right |
| **Pinch Spread** | Zoom in | Spreading thumb-index apart triggers zoom in |
| **Pinch Close** | Zoom out | Closing thumb-index together triggers zoom out |
| **Two-Finger Scroll** | Scroll | Vertical two-finger motion scrolls the page |
| **Thumbs Up** | Confirm | Thumbs up gesture confirms the current action |

All gesture-to-action mappings are fully customizable in `config/gesture_mappings.yaml`, including per-application overrides (e.g., swipe left/right becomes previous/next slide in PowerPoint).

---

## Supported Voice Commands

Horizon UI ships with 30+ built-in voice commands organized by category:

### Mouse and Navigation
| Command | Action |
|---|---|
| "click" / "select" / "press" | Left click |
| "right click" / "context menu" | Right click |
| "scroll up" / "page up" | Scroll up |
| "scroll down" / "page down" | Scroll down |
| "zoom in" / "enlarge" | Zoom in |
| "zoom out" / "shrink" | Zoom out |

### Application Control
| Command | Action |
|---|---|
| "open {app}" / "launch {app}" | Open application by name |
| "close" / "close window" | Close window (Alt+F4) |
| "minimize" | Minimize window |
| "maximize" / "fullscreen" | Maximize window |
| "switch window" / "alt tab" | Switch window (Alt+Tab) |

### Text and Clipboard
| Command | Action |
|---|---|
| "type {text}" / "write {text}" | Type arbitrary text |
| "copy" / "copy that" | Copy (Ctrl+C) |
| "paste" | Paste (Ctrl+V) |
| "undo" | Undo (Ctrl+Z) |

### Presentation
| Command | Action |
|---|---|
| "next slide" | Next slide (Right arrow) |
| "previous slide" | Previous slide (Left arrow) |

### Media and System
| Command | Action |
|---|---|
| "volume up" / "louder" | Volume up |
| "volume down" / "softer" | Volume down |
| "mute" / "mute microphone" | Toggle mute |
| "brightness up" / "brighter" | Brightness up |
| "brightness down" / "dimmer" | Brightness down |
| "screenshot" / "capture screen" | Take screenshot |
| "lock" / "lock screen" | Lock workstation |

### System Control
| Command | Action |
|---|---|
| "pause" / "stop listening" | Pause voice recognition |
| "resume" / "start listening" | Resume voice recognition |
| "help" / "show commands" | Show help overlay |
| "calibrate" / "recalibrate" | Start calibration wizard |
| "sterile mode" / "clean mode" | Activate sterile profile |

All commands are defined in `config/voice_commands.yaml` and can be customized, extended, or overridden.

---

## System Requirements

### Hardware
- **CPU**: Any modern x86-64 processor (Intel i5 / AMD Ryzen 5 or better recommended)
- **RAM**: 4 GB minimum, 8 GB recommended
- **Webcam**: Any USB or built-in webcam (640x480 or higher resolution)
- **Microphone**: Any input audio device (built-in or external)
- **GPU**: Not required (all inference runs on CPU via ONNX Runtime)

### Software
- **Operating System**: Windows 10 (version 1903+) or Windows 11
- **Python**: 3.11 or later
- **pip**: 21.0 or later

### Runtime Resource Budgets
| Resource | Target |
|---|---|
| CPU usage | < 50% of total (sustained) |
| RAM usage | < 3.5 GB |
| Gesture latency (end-to-end) | < 120ms |
| Voice latency (end-to-end) | < 500ms |
| Gesture classification F1 score | >= 0.85 |
| Voice intent accuracy | >= 90% |

---

## Installation

### Standard Installation

```bash
# Clone or download the project
cd horizon-ui

# Install the package and all dependencies
pip install -e .
```

### Development Installation

```bash
# Install with development tools (pytest, ruff, mypy)
pip install -e ".[dev]"
```

### ML Training Installation

```bash
# Install with PyTorch for gesture model training
pip install -e ".[training]"
```

### Dependency Overview

| Package | Purpose |
|---|---|
| `opencv-python` | Webcam frame capture and image processing |
| `mediapipe` | Hand detection and 21-point landmark estimation |
| `onnxruntime` | Gesture classifier inference (GRU/1D-CNN ONNX model) |
| `numpy` | Numerical computation, array operations |
| `PyQt6` | Transparent overlay window, settings UI, system tray |
| `sounddevice` | Microphone audio capture (PortAudio wrapper) |
| `webrtcvad` | Voice activity detection (speech onset/offset) |
| `faster-whisper` | On-device speech recognition (CTranslate2 Whisper) |
| `vosk` | Alternative on-device ASR (Kaldi-based) |
| `pyyaml` | YAML configuration file parsing |
| `pywin32` | Windows Named Pipes for plugin IPC |
| `pynput` | Keyboard/mouse monitoring utilities |

---

## Usage

### Running the Application

```bash
# Standard launch
python -m horizon

# Or via the installed command
horizon
```

### Command-Line Options

```
usage: horizon [-h] [--config PATH] [--profile {sterile,low_vision,voice_only}]
               [--log-level {DEBUG,INFO,WARNING,ERROR}] [--no-overlay]

Horizon UI -- Multimodal Desktop Overlay for Touchless Control

options:
  -h, --help            Show this help message and exit
  --config PATH         Path to a custom settings YAML file (overrides defaults)
  --profile NAME        Activate an accessibility profile on startup
                        Choices: sterile, low_vision, voice_only
  --log-level LEVEL     Set the logging verbosity
                        Choices: DEBUG, INFO, WARNING, ERROR (default: INFO)
  --no-overlay          Run in headless mode without the overlay UI
                        (useful for testing or background operation)
```

### Examples

```bash
# Launch with the sterile accessibility profile
python -m horizon --profile sterile

# Launch with verbose logging for debugging
python -m horizon --log-level DEBUG

# Launch headless (no overlay window, processing only)
python -m horizon --no-overlay

# Launch with a custom configuration file
python -m horizon --config my_settings.yaml
```

### Global Hotkey

Press **Ctrl+Shift+H** at any time to toggle Horizon UI between active and paused states. This hotkey is configurable in `config/default_settings.yaml`.

### System Tray

When running, Horizon UI places an icon in the Windows system tray. Right-click the tray icon for:

- **Pause / Resume** -- Toggle tracking on/off
- **Settings** -- Open the settings dialog
- **Calibrate** -- Launch the calibration wizard
- **About** -- Show version information
- **Quit** -- Gracefully shut down

### Quick Demo (No ML Model Required)

To test hand tracking without the trained gesture classifier:

```bash
python scripts/demo.py
```

This launches a simple OpenCV window with MediaPipe hand detection and basic rule-based gesture recognition.

---

## Configuration

All configuration is stored in YAML files under the `config/` directory. Settings follow a layered precedence model:

```
Schema Defaults (code)
  +-- config/default_settings.yaml (project defaults)
      +-- %APPDATA%/HorizonUI/settings.yaml (user overrides)
          +-- Active profile overrides (accessibility)
              +-- CLI arguments (highest priority)
```

### Core Settings (`config/default_settings.yaml`)

```yaml
camera:
  device_index: 0              # Webcam device index (0 = default)
  resolution: [640, 480]       # Capture resolution [width, height]
  fps: 30                      # Target frames per second

audio:
  sample_rate: 16000           # Audio sample rate in Hz (16kHz for ASR)
  channels: 1                  # Mono audio
  chunk_duration_ms: 30        # Audio chunk size for VAD processing

mediapipe:
  max_num_hands: 1             # Maximum number of hands to track
  min_detection_confidence: 0.7 # Minimum confidence for hand detection
  min_tracking_confidence: 0.6  # Minimum confidence for hand tracking

asr:
  backend: "whisper"           # ASR engine: "whisper" or "vosk"
  whisper_model: "small"       # Whisper model size: tiny, base, small, medium
  language: "en"               # Recognition language

gesture:
  confidence_threshold: 0.75   # Minimum confidence to accept a gesture
  smoothing_alpha: 0.3         # EMA smoothing factor (0=smooth, 1=responsive)
  temporal_window_frames: 15   # Number of frames in the feature buffer

overlay:
  opacity: 0.85                # Overlay opacity (0.0 - 1.0)
  theme: "dark"                # Theme: dark, light, high_contrast
  show_hand_skeleton: true     # Draw the hand skeleton on the overlay
  cursor_size: 24              # Virtual cursor diameter in pixels

system:
  startup_with_os: false       # Auto-start with Windows
  global_hotkey: "Ctrl+Shift+H" # Toggle hotkey
  log_level: "INFO"            # Logging level
```

### Gesture Mappings (`config/gesture_mappings.yaml`)

Defines which action each gesture triggers, with optional per-application overrides:

```yaml
gestures:
  pinch:
    action: left_click
    description: "Thumb-index pinch triggers left click"
  # ... more gestures

app_overrides:
  "PowerPoint":
    swipe_left:
      action: key_press
      key: "left"              # Swipe left = previous slide in PowerPoint
    swipe_right:
      action: key_press
      key: "right"             # Swipe right = next slide in PowerPoint
```

### Voice Commands (`config/voice_commands.yaml`)

Defines the voice command vocabulary with pattern matching and parameters:

```yaml
commands:
  - pattern: ["open {app}", "launch {app}", "start {app}"]
    action: open_application
    params: [app]              # {app} is extracted as a parameter

  - pattern: ["copy", "copy that"]
    action: key_combo
    keys: ["ctrl", "c"]       # Static key combination
```

---

## Accessibility Profiles

Profiles are YAML files in `config/profiles/` that override specific settings for accessibility needs. Activate a profile via CLI (`--profile sterile`), voice command ("sterile mode"), or the settings panel.

### Sterile Mode (`sterile.yaml`)
Designed for touchless operation in sterile environments such as operating rooms:
- Higher gesture confidence threshold (0.85) to reduce false positives
- Increased cursor smoothing for steadier control
- High-contrast theme with larger cursor (32px)
- Audio feedback enabled for all actions

### Low Vision Mode (`low_vision.yaml`)
Optimized for users with visual impairments (WCAG 2.1 AA compliant):
- Extra-large cursor (48px diameter)
- High-contrast theme with maximum opacity (1.0)
- Hand skeleton always visible
- Verbose audio cues for all state changes

### Voice-Only Mode (`voice_only.yaml`)
Disables all camera-based input for environments where gesture control is unavailable:
- Camera and gesture tracking disabled
- Voice confirmation for all actions
- Audio feedback enabled
- No hand skeleton or cursor display

### Creating Custom Profiles

Create a new YAML file in `config/profiles/`:

```yaml
name: "My Custom Profile"
description: "Description of this profile"

overrides:
  gesture:
    confidence_threshold: 0.9
  overlay:
    theme: "light"
    cursor_size: 32
  system:
    audio_feedback: true
```

---

## Plugin System

Horizon UI supports external plugins that communicate via **JSON-RPC 2.0** over **Windows Named Pipes**.

### Built-in Plugins

| Plugin | Description |
|---|---|
| **Media Control** | Play/pause, next/previous track, volume control via system media keys |
| **Slide Control** | Next/previous slide, start/end slideshow for presentation software |

### Plugin Architecture

```
Horizon UI (Main Process)
    |
    +-- PluginManager (discovers, loads, manages lifecycle)
    |
    +-- IPCServer (Named Pipe: \\.\pipe\HorizonUI)
    |   +-- JSON-RPC 2.0 message handling
    |
    +-- IPCAuth (token-based authentication)
        +-- Each plugin gets a unique SHA-256 token

Plugin Process
    |
    +-- Connects to Named Pipe -> Authenticates -> Receives events -> Sends responses
```

### Writing a Plugin

Implement the `BasePlugin` abstract class:

```python
from horizon.plugins.base_plugin import BasePlugin

class MyPlugin(BasePlugin):
    def get_manifest(self):
        return {
            "name": "My Plugin",
            "version": "1.0.0",
            "description": "What this plugin does",
            "author": "Your Name",
            "permissions": ["input_injection"],
        }

    def on_activate(self):
        # Called when the plugin is activated
        pass

    def on_deactivate(self):
        # Called when the plugin is deactivated
        pass

    def on_event(self, event_type, data):
        # Called when a relevant event occurs
        if data.get("action") == "my_action":
            # Handle the action
            return {"status": "ok"}
        return None
```

---

## Project Structure

```
horizon-ui/
|-- pyproject.toml                          # Build configuration and dependencies
|-- README.md                               # This file
|-- .gitignore                              # Git ignore rules
|-- .github/
|   +-- workflows/
|       +-- ci.yml                          # GitHub Actions CI pipeline
|
|-- config/                                 # YAML configuration files
|   |-- default_settings.yaml               # Core application settings
|   |-- gesture_mappings.yaml               # Gesture-to-action mappings
|   |-- voice_commands.yaml                 # Voice command definitions
|   +-- profiles/                           # Accessibility profiles
|       |-- sterile.yaml                    # Sterile environment mode
|       |-- low_vision.yaml                 # Low-vision accessibility
|       +-- voice_only.yaml                 # Voice-only operation
|
|-- models/                                 # ML model files
|   +-- README.md                           # Model download/training instructions
|
|-- src/horizon/                            # Main application package
|   |-- __init__.py                         # Package init, version
|   |-- __main__.py                         # Entry point (CLI argument parsing)
|   |-- app.py                              # Application orchestrator
|   |-- event_bus.py                        # Central pub/sub event bus
|   |-- types.py                            # Shared dataclasses and enums
|   |-- constants.py                        # Global constants and thresholds
|   |
|   |-- capture/                            # Layer 1: Input Capture
|   |   |-- frame_capture.py                # OpenCV webcam capture thread
|   |   |-- audio_capture.py                # sounddevice microphone stream
|   |   +-- device_manager.py               # Camera/mic enumeration and health
|   |
|   |-- perception/                         # Layer 2: Signal Processing
|   |   |-- hand_detector.py                # MediaPipe Hands wrapper
|   |   |-- landmark_estimator.py           # 21-point landmark extraction
|   |   |-- feature_extractor.py            # Temporal feature computation
|   |   |-- gesture_classifier.py           # ONNX Runtime GRU/1D-CNN inference
|   |   |-- smoothing.py                    # EMA + Kalman cursor smoothing
|   |   |-- coordinate_mapper.py            # Camera-to-screen coordinate mapping
|   |   |-- vad.py                          # WebRTC voice activity detection
|   |   |-- asr.py                          # ASR backends (Whisper + Vosk)
|   |   +-- intent_parser.py                # Voice transcript-to-intent parsing
|   |
|   |-- decision/                           # Layer 3: Decision Making
|   |   |-- fusion_engine.py                # Multimodal gesture+voice fusion
|   |   |-- policy_engine.py                # Confidence gating and cooldowns
|   |   |-- action_mapper.py                # Intent-to-OS action resolution
|   |   +-- context_manager.py              # Active window detection
|   |
|   |-- control/                            # Layer 4: System Control
|   |   |-- os_event_injector.py            # Win32 SendInput (mouse/keyboard)
|   |   |-- platform_abstract.py            # OS abstraction layer
|   |   |-- plugin_manager.py               # Plugin lifecycle management
|   |   |-- ipc_server.py                   # JSON-RPC 2.0 Named Pipe server
|   |   +-- ipc_auth.py                     # Token-based plugin authentication
|   |
|   |-- presentation/                       # Layer 5: User Interface
|   |   |-- overlay_ui.py                   # PyQt6 transparent overlay window
|   |   |-- hud_renderer.py                 # HUD element rendering
|   |   |-- settings_panel.py               # Settings dialog (tabbed)
|   |   |-- audio_feedback.py               # Sound cue player
|   |   |-- tray_icon.py                    # System tray icon and menu
|   |   |-- calibration_wizard.py           # 4-point calibration flow
|   |   +-- themes.py                       # Theme definitions
|   |
|   |-- profiles/                           # Settings Management
|   |   |-- profile_manager.py              # Profile loading and switching
|   |   |-- settings_store.py               # YAML settings with layered merge
|   |   +-- schema.py                       # Settings validation schema
|   |
|   +-- plugins/                            # Built-in Plugins
|       |-- base_plugin.py                  # Abstract plugin base class
|       |-- media_control.py                # Media playback plugin
|       +-- slide_control.py                # Presentation slide plugin
|
|-- training/                               # Offline ML training pipeline
|   |-- README.md                           # Training instructions
|   |-- requirements.txt                    # Training-specific dependencies
|   |-- dataset/                            # Training data directory
|   |   +-- README.md                       # Dataset format documentation
|   |-- train_gesture_model.py              # PyTorch GRU/1D-CNN training
|   |-- export_onnx.py                      # PyTorch-to-ONNX export + quantization
|   |-- evaluate_model.py                   # Metrics: F1, confusion matrix
|   +-- augment_data.py                     # Data augmentation (noise, warp, mirror)
|
|-- tests/                                  # Test suite
|   |-- conftest.py                         # Shared fixtures
|   |-- unit/                               # 20 unit test files
|   |   |-- test_frame_capture.py
|   |   |-- test_audio_capture.py
|   |   |-- test_hand_detector.py
|   |   |-- test_landmark_estimator.py
|   |   |-- test_feature_extractor.py
|   |   |-- test_gesture_classifier.py
|   |   |-- test_smoothing.py
|   |   |-- test_coordinate_mapper.py
|   |   |-- test_vad.py
|   |   |-- test_asr.py
|   |   |-- test_intent_parser.py
|   |   |-- test_fusion_engine.py
|   |   |-- test_policy_engine.py
|   |   |-- test_action_mapper.py
|   |   |-- test_context_manager.py
|   |   |-- test_os_event_injector.py
|   |   |-- test_plugin_manager.py
|   |   |-- test_ipc_server.py
|   |   |-- test_profile_manager.py
|   |   +-- test_settings_store.py
|   |-- integration/                        # End-to-end pipeline tests
|   |   |-- test_gesture_pipeline.py
|   |   |-- test_voice_pipeline.py
|   |   |-- test_fusion_pipeline.py
|   |   +-- test_plugin_system.py
|   +-- performance/                        # Latency and resource benchmarks
|       |-- test_latency.py
|       +-- test_resource_usage.py
|
|-- scripts/                                # Utility scripts
|   |-- collect_gesture_data.py             # Webcam data collection tool
|   |-- benchmark.py                        # Performance benchmark runner
|   +-- demo.py                             # Quick hand tracking demo
|
|-- docs/
|   +-- architecture.md                     # Architecture documentation
|
+-- assets/
    |-- icons/                              # Application icons
    |   |-- tray_icon.png
    |   +-- app_icon.ico
    +-- sounds/                             # Audio feedback cues
        |-- activate.wav
        |-- deactivate.wav
        +-- error.wav
```

---

## Technology Stack

| Component | Technology | Version |
|---|---|---|
| **Language** | Python | 3.11+ |
| **Camera Capture** | OpenCV (`cv2.VideoCapture`) | >= 4.8 |
| **Hand Detection** | MediaPipe Hands (21-point landmarks) | >= 0.10 |
| **Gesture Classification** | ONNX Runtime (GRU / 1D-CNN) | >= 1.16 |
| **Numerical Computing** | NumPy | >= 1.24 |
| **Overlay UI** | PyQt6 (transparent frameless window) | >= 6.5 |
| **Audio Capture** | sounddevice (PortAudio) | >= 0.4 |
| **Voice Activity Detection** | webrtcvad | >= 2.0.10 |
| **Speech Recognition (primary)** | faster-whisper (CTranslate2) | >= 0.10 |
| **Speech Recognition (fallback)** | Vosk (Kaldi) | >= 0.3.45 |
| **Configuration** | PyYAML | >= 6.0 |
| **Windows Integration** | pywin32 (Named Pipes), ctypes (SendInput) | >= 306 |
| **Input Monitoring** | pynput | >= 1.7 |
| **ML Training** | PyTorch, scikit-learn, matplotlib | >= 2.1 |
| **Testing** | pytest, pytest-cov, pytest-qt, pytest-mock | >= 7.4 |
| **Linting** | Ruff | >= 0.1 |
| **Type Checking** | mypy (strict mode) | >= 1.6 |
| **Build System** | hatchling (via pyproject.toml) | latest |
| **CI/CD** | GitHub Actions (Windows runner) | -- |

---

## ML Training Pipeline

The gesture classifier is a temporal neural network (GRU or 1D-CNN) that classifies sequences of hand landmark features into one of 11 gesture classes.

### Pipeline Overview

```
1. Data Collection    ->  scripts/collect_gesture_data.py
   Record webcam sessions per gesture class

2. Data Augmentation  ->  training/augment_data.py
   Apply noise, scaling, time warping, and mirroring

3. Model Training     ->  training/train_gesture_model.py
   Train GRU or 1D-CNN on landmark sequences with PyTorch

4. ONNX Export        ->  training/export_onnx.py
   Convert PyTorch model to ONNX with optional INT8 quantization

5. Evaluation         ->  training/evaluate_model.py
   Generate classification report, confusion matrix, F1 scores

6. Deployment         ->  Copy .onnx file to models/gesture_classifier.onnx
```

### Training Data Format

Each sample is a NumPy array of shape `(T, 63)`:
- **T** = number of frames (variable, typically 30-150 depending on gesture duration)
- **63** = 21 MediaPipe landmarks x 3 coordinates (x, y, z), normalized to [0, 1]

### Model Architectures

**GRU (default):**
- 2-layer GRU with 128 hidden units
- Classifier: Linear(128, 64) -> ReLU -> Dropout(0.3) -> Linear(64, 11)
- Uses the last timestep output for classification

**1D-CNN (alternative):**
- Conv1d(63, 64, k=3) -> BatchNorm -> ReLU -> Conv1d(64, 128, k=3) -> BatchNorm -> ReLU -> AdaptiveAvgPool
- Classifier: Linear(128, 64) -> ReLU -> Dropout(0.3) -> Linear(64, 11)

### Quick Start

```bash
cd training
pip install -r requirements.txt

# Collect 20 samples of "pinch" gesture
python ../scripts/collect_gesture_data.py --gesture pinch --output dataset/raw/pinch/ --samples 20

# Train the model
python train_gesture_model.py --data dataset/processed --model gru --epochs 50 --batch-size 32

# Export to ONNX with INT8 quantization
python export_onnx.py --checkpoint checkpoints/best_model.pt --output ../models/gesture_classifier.onnx --quantize

# Evaluate
python evaluate_model.py --model ../models/gesture_classifier.onnx --data dataset/processed
```

---

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run performance tests
pytest tests/performance/

# Run with coverage report
pytest --cov=horizon --cov-report=html
```

### Test Categories

| Category | Count | Purpose |
|---|---|---|
| **Unit tests** | 20 files | Test each module in isolation with mocked hardware |
| **Integration tests** | 4 files | End-to-end pipeline tests (gesture, voice, fusion, plugins) |
| **Performance tests** | 2 files | Latency and memory usage verification against SRS targets |

### Linting and Type Checking

```bash
# Lint with Ruff
ruff check src/ tests/

# Format check
ruff format --check src/ tests/

# Type check with mypy (strict mode)
mypy src/horizon/ --ignore-missing-imports
```

### Performance Benchmarks

```bash
# Run the benchmark suite
python scripts/benchmark.py

# Example output:
# Horizon UI Performance Benchmarks
# ==================================================
#   EMA Filter                    : 0.0012 ms/op
#   Kalman Filter 2D              : 0.0089 ms/op
#   Feature Extraction            : 1.2340 ms/op
#   EventBus Publish (3 subs)     : 0.0034 ms/op
```

---

## Performance Targets

These targets are derived from the SRS document and verified by the test suite:

| Metric | Target | Measurement Method |
|---|---|---|
| Gesture end-to-end latency | < 120ms | Frame capture to OS event injection |
| Voice end-to-end latency | < 500ms | Speech offset to OS event injection |
| CPU usage (sustained) | < 50% | Total system CPU during active operation |
| RAM usage | < 3.5 GB | Peak resident memory |
| Gesture classification F1 | >= 0.85 | Weighted F1 on held-out test set |
| Voice intent accuracy | >= 90% | Correct intent on test transcript set |
| Frame throughput | >= 25 FPS | Processed frames per second |
| EMA filter latency | < 1ms | Per-call processing time |
| EventBus publish latency | < 1ms | Per-event dispatch with 3 subscribers |
| Feature extraction latency | < 10ms | Per-frame feature computation |

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Install development dependencies: `pip install -e ".[dev]"`
4. Make your changes and add tests
5. Run the test suite: `pytest`
6. Run linting: `ruff check src/ tests/`
7. Run type checking: `mypy src/horizon/ --ignore-missing-imports`
8. Commit and push your branch
9. Open a pull request

---

## License

MIT License. See [LICENSE](LICENSE) for details.
