# Models

## Gesture Classifier

Place the trained ONNX gesture classifier here as `gesture_classifier.onnx`.

### Training

To train and export the model:

```bash
cd training
pip install -r requirements.txt

# Collect data
python ../scripts/collect_gesture_data.py --gesture pinch --output dataset/raw/pinch/ --samples 20

# Train
python train_gesture_model.py --data dataset/processed --epochs 50

# Export to ONNX
python export_onnx.py --checkpoint checkpoints/best_model.pt --output ../models/gesture_classifier.onnx --quantize
```

## ASR Models

### Whisper (faster-whisper)

Models are downloaded automatically on first use by the `faster-whisper` library.
Default model: `small` (~462MB, downloaded to cache).

### Vosk

Download a Vosk model from https://alphacephei.com/vosk/models
and extract to a directory. Specify the path in settings:

```yaml
asr:
  backend: "vosk"
  vosk_model_path: "path/to/vosk-model-small-en-us-0.15"
```
