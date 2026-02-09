"""Generate voice recognition metrics: confusion matrix, precision, recall, F1, latency."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from horizon.event_bus import EventBus
from horizon.perception.intent_parser import IntentParser

# ---------------------------------------------------------------------------
# Test dataset: simulated ASR transcripts mapped to expected actions
# Includes exact, partial, fuzzy, noisy, and negative samples
# ---------------------------------------------------------------------------

# Each entry: (transcript_from_ASR, expected_action_or_None)
TEST_SAMPLES = [
    # --- Exact matches (clean ASR output) ---
    ("click", "left_click"),
    ("select", "left_click"),
    ("press", "left_click"),
    ("right click", "right_click"),
    ("context menu", "right_click"),
    ("double click", "double_click"),
    ("double tap", "double_click"),
    ("scroll up", "scroll_up"),
    ("scroll down", "scroll_down"),
    ("page up", "scroll_up"),
    ("page down", "scroll_down"),
    ("copy", "key_combo"),
    ("copy that", "key_combo"),
    ("paste", "key_combo"),
    ("undo", "key_combo"),
    ("close", "key_combo"),
    ("close window", "key_combo"),
    ("minimize", "minimize_window"),
    ("minimize window", "minimize_window"),
    ("maximize", "maximize_window"),
    ("maximize window", "maximize_window"),
    ("fullscreen", "maximize_window"),
    ("switch window", "key_combo"),
    ("alt tab", "key_combo"),
    ("next slide", "key_press"),
    ("previous slide", "key_press"),
    ("zoom in", "zoom_in"),
    ("zoom out", "zoom_out"),
    ("enlarge", "zoom_in"),
    ("shrink", "zoom_out"),
    ("mute", "toggle_mic_mute"),
    ("mute microphone", "toggle_mic_mute"),
    ("pause", "pause_voice"),
    ("stop listening", "pause_voice"),
    ("resume", "resume_voice"),
    ("start listening", "resume_voice"),
    ("brightness up", "brightness_up"),
    ("brighter", "brightness_up"),
    ("brightness down", "brightness_down"),
    ("dimmer", "brightness_down"),
    ("volume up", "volume_up"),
    ("louder", "volume_up"),
    ("volume down", "volume_down"),
    ("softer", "volume_down"),
    ("screenshot", "screenshot"),
    ("capture screen", "screenshot"),
    ("lock", "lock_screen"),
    ("lock screen", "lock_screen"),
    ("help", "show_help_overlay"),
    ("show commands", "show_help_overlay"),
    ("calibrate", "start_calibration"),
    ("recalibrate", "start_calibration"),
    ("sterile mode", "activate_profile"),
    ("clean mode", "activate_profile"),

    # --- Parameterized commands ---
    ("open notepad", "open_application"),
    ("open chrome", "open_application"),
    ("launch calculator", "open_application"),
    ("start explorer", "open_application"),
    ("type hello world", "type_text"),
    ("write meeting notes", "type_text"),

    # --- Fuzzy / noisy ASR transcripts (common Whisper errors) ---
    ("clik", "left_click"),                   # typo: missing 'c'
    ("dubble click", "double_click"),         # ASR phonetic
    ("scrol up", "scroll_up"),               # missing letter
    ("scrol down", "scroll_down"),           # missing letter
    ("zooom in", "zoom_in"),                 # extra letter
    ("rite click", "right_click"),           # phonetic spelling
    ("coopy", "key_combo"),                  # typo
    ("passte", "key_combo"),                 # extra letter
    ("undu", "key_combo"),                   # typo
    ("nexxt slide", "key_press"),            # extra letter
    ("previous slid", "key_press"),          # missing letter
    ("minemize", "minimize_window"),         # typo
    ("maximize windoe", "maximize_window"),  # typo
    ("volumme up", "volume_up"),             # extra letter
    ("mewt", "toggle_mic_mute"),             # phonetic (hard)
    ("screnshot", "screenshot"),             # missing letter
    ("lok screen", "lock_screen"),           # missing letter
    ("calibrait", "start_calibration"),      # phonetic
    ("brightnes up", "brightness_up"),       # missing letter

    # --- Realistic Whisper ASR errors (hallucinations, filler words, partial) ---
    ("um click", "left_click"),               # filler word prefix
    ("scroll scroll down", "scroll_down"),    # stutter/repeat
    ("can you zoom in", "zoom_in"),           # conversational phrasing
    ("please close window", "key_combo"),     # polite prefix
    ("uh right click", "right_click"),        # filler word
    ("go to next slide", "key_press"),        # extra context
    ("i want to copy", "key_combo"),          # conversational
    ("make it bigger", "zoom_in"),            # synonym (indirect)
    ("make it smaller", "zoom_out"),          # synonym (indirect)
    ("take a screenshot", "screenshot"),      # extra words
    ("open the chrome", "open_application"),  # extra article
    ("type the word hello", "type_text"),     # extra context
    ("go back", "key_press"),                 # ambiguous (prev slide?)
    ("switch to window", "key_combo"),        # rephrased
    ("turn up volume", "volume_up"),          # reordered words
    ("turn down volume", "volume_down"),      # reordered words
    ("screen lock", "lock_screen"),           # word order swapped
    ("screen capture", "screenshot"),         # synonym
    ("full screen mode", "maximize_window"),  # rephrased
    ("minimize it", "minimize_window"),       # extra pronoun
    ("scroll a bit down", "scroll_down"),     # extra words in middle
    ("do a double click", "double_click"),    # extra prefix words
    ("paste it here", "key_combo"),           # extra suffix
    ("undo that", "key_combo"),              # extra word
    ("show me the commands", "show_help_overlay"),  # rephrased
    ("start sterile mode", "activate_profile"),     # extra prefix
    ("pause listening", "pause_voice"),              # merged pattern
    ("just click", "left_click"),                    # filler prefix
    ("click on that", "left_click"),                 # extra suffix
    ("volume louder", "volume_up"),                  # mixed patterns

    # --- Hard negatives (look like commands but aren't) ---
    ("i clicked on the button", None),        # past tense - narration
    ("the scroll bar is broken", None),       # narration about scroll
    ("i need to zoom the camera", None),      # different context
    ("let me copy this paragraph", None),     # narration
    ("she locked herself out", None),         # narrative context
    ("the volume was too loud", None),        # past tense description
    ("do you like the screenshot", None),     # question about screenshot
    ("open ended question", None),            # has "open" but not a command
    ("close enough", None),                   # has "close" but idiom
    ("paste the sticker on the wall", None),  # different meaning of paste

    # --- Pure noise / irrelevant (should NOT match) ---
    ("hello there", None),
    ("what time is it", None),
    ("how are you", None),
    ("the weather is nice", None),
    ("random noise", None),
    ("testing one two three", None),
    ("abcdef", None),
    ("i like pizza", None),
    ("this is a sentence", None),
    ("what is the meaning of life", None),
    ("computer says no", None),
    ("let me think about it", None),
    ("good morning everyone", None),
    ("thank you very much", None),
    ("see you later", None),
]

# Command categories for grouping
COMMAND_CATEGORIES = {
    "left_click": "Click/Select",
    "right_click": "Click/Select",
    "double_click": "Click/Select",
    "scroll_up": "Navigation",
    "scroll_down": "Navigation",
    "key_combo": "Keyboard",
    "key_press": "Keyboard",
    "minimize_window": "Window Mgmt",
    "maximize_window": "Window Mgmt",
    "zoom_in": "Zoom",
    "zoom_out": "Zoom",
    "toggle_mic_mute": "Voice Control",
    "pause_voice": "Voice Control",
    "resume_voice": "Voice Control",
    "brightness_up": "System",
    "brightness_down": "System",
    "volume_up": "System",
    "volume_down": "System",
    "screenshot": "System",
    "lock_screen": "System",
    "show_help_overlay": "Accessibility",
    "start_calibration": "Accessibility",
    "activate_profile": "Accessibility",
    "open_application": "App Control",
    "type_text": "Text Input",
    None: "No Command",
}


def main() -> None:
    output_dir = str(project_root / "metrics_output")
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the intent parser
    bus = EventBus()
    parser = IntentParser(event_bus=bus)
    print(f"IntentParser loaded with {len(parser._commands)} commands")
    print()

    # ---- Run all test samples ----
    print("=" * 70)
    print("VOICE COMMAND RECOGNITION EVALUATION")
    print("=" * 70)

    y_true = []       # expected action string or "none"
    y_pred = []       # predicted action string or "none"
    latencies = []    # parsing latency per sample
    results_detail = []

    for transcript, expected_action in TEST_SAMPLES:
        expected_label = expected_action if expected_action else "none"

        start = time.perf_counter()
        intent = parser.parse(transcript)
        elapsed_ms = (time.perf_counter() - start) * 1000

        predicted_label = intent.action.value if intent else "none"
        correct = predicted_label == expected_label

        y_true.append(expected_label)
        y_pred.append(predicted_label)
        latencies.append(elapsed_ms)

        results_detail.append({
            "transcript": transcript,
            "expected": expected_label,
            "predicted": predicted_label,
            "confidence": intent.confidence if intent else 0.0,
            "correct": correct,
            "latency_ms": round(elapsed_ms, 4),
        })

    # ---- Compute metrics ----
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        accuracy_score,
    )

    # All unique labels
    all_labels = sorted(set(y_true) | set(y_pred))

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    report_str = classification_report(
        y_true, y_pred,
        labels=all_labels,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=all_labels)

    # ---- Print results ----
    print()
    print(f"Test samples: {len(TEST_SAMPLES)}")
    positive_count = sum(1 for t, a in TEST_SAMPLES if a is not None)
    negative_count = sum(1 for t, a in TEST_SAMPLES if a is None)
    exact_count = 60   # exact pattern matches
    fuzzy_count = 19   # typo/phonetic errors
    asr_noisy = positive_count - exact_count - fuzzy_count  # realistic ASR noise
    print(f"  Exact transcripts:  {exact_count}")
    print(f"  Fuzzy/typo:         {fuzzy_count}")
    print(f"  Realistic ASR noise:{asr_noisy}")
    print(f"  Negative samples:   {negative_count}")
    print()

    print("=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    print(report_str)

    print(f"Overall Accuracy:  {accuracy:.4f}")
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall:    {recall:.4f}")
    print(f"Weighted F1 Score:  {f1:.4f}")
    print()

    # ---- Confusion Matrix ----
    print("=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    # Print condensed confusion matrix
    short_labels = []
    for label in all_labels:
        if len(label) > 10:
            short_labels.append(label[:9] + ".")
        else:
            short_labels.append(label)

    max_w = max(len(s) for s in short_labels) + 1
    header = " " * (max_w + 2) + "  ".join(f"{s:>8s}" for s in short_labels)
    print(header)
    print(" " * (max_w + 2) + "-" * (len(short_labels) * 10))
    for i, row in enumerate(cm):
        row_str = f"{short_labels[i]:>{max_w}s} |"
        for val in row:
            row_str += f" {val:7d} "
        row_str += f"| {sum(row):4d}"
        print(row_str)
    print()

    # ---- Per-category accuracy ----
    print("=" * 70)
    print("PER-CATEGORY ACCURACY")
    print("=" * 70)
    category_correct = {}
    category_total = {}
    for i, (transcript, expected) in enumerate(TEST_SAMPLES):
        cat = COMMAND_CATEGORIES.get(expected, "Unknown")
        if cat not in category_correct:
            category_correct[cat] = 0
            category_total[cat] = 0
        category_total[cat] += 1
        if results_detail[i]["correct"]:
            category_correct[cat] += 1

    for cat in sorted(category_total.keys()):
        total = category_total[cat]
        correct = category_correct[cat]
        acc = correct / total if total > 0 else 0
        print(f"  {cat:20s}: {correct:3d}/{total:3d} ({acc:.1%})")
    print()

    # ---- Misclassified samples ----
    print("=" * 70)
    print("MISCLASSIFIED SAMPLES")
    print("=" * 70)
    misclassified = [r for r in results_detail if not r["correct"]]
    if misclassified:
        for r in misclassified:
            print(f'  "{r["transcript"]}" -> expected: {r["expected"]}, got: {r["predicted"]} (conf={r["confidence"]:.2f})')
    else:
        print("  None! All samples classified correctly.")
    print()

    # ---- Latency ----
    print("=" * 70)
    print("LATENCY BENCHMARKS")
    print("=" * 70)
    avg_lat = np.mean(latencies)
    p95_lat = np.percentile(latencies, 95)
    max_lat = np.max(latencies)
    print(f"  Average parsing latency: {avg_lat:.4f} ms")
    print(f"  P95 parsing latency:     {p95_lat:.4f} ms")
    print(f"  Max parsing latency:     {max_lat:.4f} ms")
    print(f"  Target:                  < 50ms per parse")
    print(f"  Status:                  {'PASS' if avg_lat < 50 else 'FAIL'}")
    print()

    # ---- Save results ----
    results = {
        "dataset": {
            "total_samples": len(TEST_SAMPLES),
            "exact_transcripts": exact_count,
            "fuzzy_typo": fuzzy_count,
            "asr_noisy": asr_noisy,
            "negative_samples": negative_count,
            "command_count": len(parser._commands),
            "pattern_count": sum(len(c.get("pattern", [])) for c in parser._commands),
        },
        "metrics": {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_weighted": round(f1, 4),
        },
        "latency": {
            "avg_ms": round(avg_lat, 4),
            "p95_ms": round(p95_lat, 4),
            "max_ms": round(max_lat, 4),
        },
        "confusion_matrix": cm.tolist(),
        "labels": all_labels,
        "per_category": {cat: {"correct": category_correct[cat], "total": category_total[cat]}
                         for cat in category_total},
        "misclassified": misclassified,
    }

    results_path = os.path.join(output_dir, "voice_metrics_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")

    # ---- Summary ----
    print()
    print("=" * 70)
    print("SUMMARY FOR PRESENTATION")
    print("=" * 70)
    print()
    print(f"Voice Commands:       {len(parser._commands)} commands, {sum(len(c.get('pattern', [])) for c in parser._commands)} patterns")
    print(f"Test Dataset:         {len(TEST_SAMPLES)} samples ({exact_count} exact + {fuzzy_count} fuzzy + {asr_noisy} ASR noisy + {negative_count} negative)")
    print(f"Accuracy:             {accuracy:.4f}")
    print(f"Precision:            {precision:.4f}")
    print(f"Recall:               {recall:.4f}")
    print(f"F1 Score:             {f1:.4f}")
    print(f"Avg Latency:          {avg_lat:.4f} ms")
    print(f"Misclassified:        {len(misclassified)}/{len(TEST_SAMPLES)}")


if __name__ == "__main__":
    main()
