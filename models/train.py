#!/usr/bin/env python3
"""
Treniranje YOLOv8 modela za detekciju biljarskih kugli.
Koristi sintetički dataset i fine-tuning na pretreniranom YOLOv8n.
"""

import os
import sys
import yaml
import json
from pathlib import Path
from datetime import datetime

# ─── Putanje ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_YAML = PROJECT_ROOT / "models" / "dataset.yaml"
MODELS_DIR   = PROJECT_ROOT / "models"
RUNS_DIR     = PROJECT_ROOT / "runs"


# ─── Konfiguracija treniranja ─────────────────────────────────────────────────

TRAIN_CONFIG = {
    "model":       "yolov8n.pt",    # nano – dobar balans brzine i točnosti
    "data":        str(DATASET_YAML),
    "epochs":      100,
    "imgsz":       640,
    "batch":       16,
    "lr0":         0.01,
    "lrf":         0.01,
    "momentum":    0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3,
    "warmup_momentum": 0.8,
    "box":         7.5,
    "cls":         0.5,
    "dfl":         1.5,
    "hsv_h":       0.015,
    "hsv_s":       0.7,
    "hsv_v":       0.4,
    "degrees":     10.0,
    "translate":   0.1,
    "scale":       0.5,
    "flipud":      0.0,
    "fliplr":      0.5,
    "mosaic":      1.0,
    "mixup":       0.1,
    "patience":    20,             # Early stopping
    "save_period": 10,
    "project":     str(RUNS_DIR),
    "name":        "billiard_detector",
    "exist_ok":    True,
    "plots":       True,
    "verbose":     True,
}


def check_dataset() -> bool:
    """Provjeri je li dataset generiran."""
    dataset_path = PROJECT_ROOT / "dataset"
    train_imgs = dataset_path / "images" / "train"
    
    if not train_imgs.exists():
        print("❌ Dataset nije pronađen!")
        print("   Pokrenite: python dataset_generator/generate_dataset.py")
        return False
    
    n_images = len(list(train_imgs.glob("*.jpg")))
    if n_images == 0:
        print("❌ Dataset je prazan!")
        return False
    
    print(f"✅ Dataset pronađen: {n_images} trening slika")
    return True


def train_model():
    """Trenira YOLOv8 model."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ ultralytics nije instaliran!")
        print("   Pokrenite: pip install ultralytics")
        sys.exit(1)

    print("🎱 Treniranje detektora biljarskih kugli")
    print("=" * 50)
    print(f"📅 Početak: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🤖 Model:   {TRAIN_CONFIG['model']}")
    print(f"📊 Epohe:   {TRAIN_CONFIG['epochs']}")
    print(f"📦 Batch:   {TRAIN_CONFIG['batch']}")
    print(f"🖼️  Veličina: {TRAIN_CONFIG['imgsz']}×{TRAIN_CONFIG['imgsz']}")
    print()

    # Provjeri dataset
    if not check_dataset():
        sys.exit(1)

    # Inicijaliziraj model
    print(f"\n📥 Učitavanje baznog modela: {TRAIN_CONFIG['model']}")
    model = YOLO(TRAIN_CONFIG["model"])

    # Treniranje
    print("\n🏋️  Početak treniranja...\n")
    results = model.train(**TRAIN_CONFIG)

    # Rezultati
    best_model = RUNS_DIR / "billiard_detector" / "weights" / "best.pt"
    last_model = RUNS_DIR / "billiard_detector" / "weights" / "last.pt"

    print("\n" + "=" * 50)
    print("✅ Treniranje završeno!")
    print(f"   💾 Najbolji model: {best_model}")
    print(f"   💾 Zadnji model:   {last_model}")

    # Spremi metriku
    metrics_path = RUNS_DIR / "billiard_detector" / "metrics.json"
    try:
        metrics = {
            "box_loss":  float(results.results_dict.get("train/box_loss", 0)),
            "cls_loss":  float(results.results_dict.get("train/cls_loss", 0)),
            "map50":     float(results.results_dict.get("metrics/mAP50(B)", 0)),
            "map50_95":  float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
            "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
            "recall":    float(results.results_dict.get("metrics/recall(B)", 0)),
        }
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n📈 Finalne metrike:")
        print(f"   mAP@0.5:    {metrics['map50']:.3f}")
        print(f"   mAP@0.5:95: {metrics['map50_95']:.3f}")
        print(f"   Preciznost: {metrics['precision']:.3f}")
        print(f"   Odziv:      {metrics['recall']:.3f}")
    except Exception as e:
        print(f"   (metrike nisu dostupne: {e})")

    print(f"\n📅 Završetak: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nSlijedeći korak:")
    print("  python evaluation/evaluate.py")
    print("  python app/detector_app.py")

    return str(best_model)


def validate_model(model_path: str = None):
    """Validacija modela na test skupu."""
    from ultralytics import YOLO

    if model_path is None:
        model_path = RUNS_DIR / "billiard_detector" / "weights" / "best.pt"

    if not Path(model_path).exists():
        print(f"❌ Model nije pronađen: {model_path}")
        return

    print(f"\n🔍 Validacija modela: {model_path}")
    model = YOLO(model_path)

    results = model.val(
        data=str(DATASET_YAML),
        split="test",
        imgsz=640,
        conf=0.25,
        iou=0.6,
    )

    print("\n📊 Rezultati validacije na test skupu:")
    print(f"   mAP@0.5:    {results.box.map50:.4f}")
    print(f"   mAP@0.5:95: {results.box.map:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Treniranje detektora biljarskih kugli")
    parser.add_argument("--validate-only", action="store_true",
                        help="Samo validiraj postojeći model")
    parser.add_argument("--model", type=str, default=None,
                        help="Putanja do modela za validaciju")
    args = parser.parse_args()

    if args.validate_only:
        validate_model(args.model)
    else:
        best = train_model()
        validate_model(best)