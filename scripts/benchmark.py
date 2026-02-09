"""Performance benchmark runner for Horizon UI."""

from __future__ import annotations

import argparse
import time

import numpy as np


def benchmark_ema(iterations: int = 10000) -> float:
    from horizon.perception.smoothing import EMAFilter

    f = EMAFilter(alpha=0.3)
    f.update(0.0, 0.0)

    start = time.perf_counter()
    for _ in range(iterations):
        f.update(np.random.rand(), np.random.rand())
    elapsed = time.perf_counter() - start
    return elapsed / iterations * 1000  # ms per iteration


def benchmark_kalman(iterations: int = 10000) -> float:
    from horizon.perception.smoothing import KalmanFilter2D

    kf = KalmanFilter2D()
    kf.update(0.0, 0.0)

    start = time.perf_counter()
    for _ in range(iterations):
        kf.update(np.random.rand(), np.random.rand())
    elapsed = time.perf_counter() - start
    return elapsed / iterations * 1000


def benchmark_feature_extraction(iterations: int = 1000) -> float:
    from horizon.event_bus import EventBus
    from horizon.perception.feature_extractor import FeatureExtractor
    from horizon.types import Landmark, LandmarkSet

    bus = EventBus()
    fe = FeatureExtractor(event_bus=bus, temporal_window=15)

    # Fill buffer
    for _ in range(15):
        lms = LandmarkSet(
            landmarks=[Landmark(x=np.random.rand(), y=np.random.rand(), z=0.0) for _ in range(21)]
        )
        fe._buffer.append(lms)

    start = time.perf_counter()
    for _ in range(iterations):
        fe._extract_features()
    elapsed = time.perf_counter() - start
    return elapsed / iterations * 1000


def benchmark_event_bus(iterations: int = 10000) -> float:
    from horizon.event_bus import EventBus
    from horizon.types import Event, EventType

    bus = EventBus()
    bus.subscribe(EventType.FRAME, lambda e: None)
    bus.subscribe(EventType.FRAME, lambda e: None)
    bus.subscribe(EventType.FRAME, lambda e: None)

    event = Event(type=EventType.FRAME, data=None)

    start = time.perf_counter()
    for _ in range(iterations):
        bus.publish(event)
    elapsed = time.perf_counter() - start
    return elapsed / iterations * 1000


def main(args: argparse.Namespace) -> None:
    print("Horizon UI Performance Benchmarks")
    print("=" * 50)

    benchmarks = [
        ("EMA Filter", benchmark_ema),
        ("Kalman Filter 2D", benchmark_kalman),
        ("Feature Extraction", benchmark_feature_extraction),
        ("EventBus Publish (3 subs)", benchmark_event_bus),
    ]

    for name, func in benchmarks:
        try:
            ms = func(args.iterations)
            print(f"  {name:30s}: {ms:.4f} ms/op")
        except Exception as e:
            print(f"  {name:30s}: ERROR - {e}")

    print("\nBenchmark complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run performance benchmarks")
    parser.add_argument("--iterations", type=int, default=10000, help="Iterations per benchmark")
    main(parser.parse_args())
