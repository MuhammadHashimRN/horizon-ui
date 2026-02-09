"""Entry point for Horizon UI — python -m horizon."""

from __future__ import annotations

import argparse
import signal
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="horizon",
        description="Horizon UI — Multimodal Desktop Overlay for Touchless Control",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom settings YAML file",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        choices=["sterile", "low_vision", "voice_only"],
        help="Activate an accessibility profile",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set log level (default: INFO)",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Run without the overlay UI (headless mode)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # PyQt6 must be imported before creating the app
    from PyQt6.QtWidgets import QApplication

    qt_app = QApplication(sys.argv)
    qt_app.setApplicationName("Horizon UI")
    qt_app.setQuitOnLastWindowClosed(False)

    from horizon.app import App

    app = App(
        config_path=args.config,
        profile=args.profile,
        log_level=args.log_level,
        no_overlay=args.no_overlay,
    )

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        app.shutdown()
        qt_app.quit()

    signal.signal(signal.SIGINT, signal_handler)

    # Start the application
    app.start()

    # Setup tray icon after app is started
    if not args.no_overlay:
        from horizon.presentation.tray_icon import TrayIcon

        tray = TrayIcon(
            parent=None,
            on_toggle=app.toggle,
            on_settings=lambda: None,  # Settings panel integration
            on_calibrate=lambda: None,  # Calibration wizard integration
            on_quit=lambda: (app.shutdown(), qt_app.quit()),
        )
        tray.show()

    # Run Qt event loop
    sys.exit(qt_app.exec())


if __name__ == "__main__":
    main()
