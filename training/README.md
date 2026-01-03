# Gesture Model Training Pipeline

## Dataset Format

Training data is stored in `dataset/` with the following structure:

```
dataset/
├── raw/          # Raw video recordings per gesture class
│   ├── point/
│   ├── pinch/
│   ├── fist/
│   └── ...
├── processed/    # Extracted landmark sequences (numpy arrays)
│   ├── point/
│   ├── pinch/
│   └── ...
└── README.md
```

Each processed file is a `.npy` file containing a (T, 63) array
where T is the number of frames and 63 = 21 landmarks * 3 coordinates (x, y, z).

## Training

```bash
pip install -r requirements.txt
python train_gesture_model.py --data dataset/processed --epochs 50 --batch-size 32
```

## Export to ONNX

```bash
python export_onnx.py --checkpoint checkpoints/best_model.pt --output ../models/gesture_classifier.onnx
```

## Evaluate

```bash
python evaluate_model.py --model ../models/gesture_classifier.onnx --data dataset/processed
```

## Data Collection

Use the collection script to record new gesture samples:

```bash
python ../scripts/collect_gesture_data.py --gesture pinch --output dataset/raw/pinch/
```
