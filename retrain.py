#!/usr/bin/env python3
import shutil
import random
from pathlib import Path
import yaml

PROJECT_ROOT = Path(".")
REAL_IMAGES  = PROJECT_ROOT / "real_dataset" / "images"
REAL_LABELS  = PROJECT_ROOT / "real_dataset" / "labels"
REAL_DATASET = PROJECT_ROOT / "real_dataset_split"
YAML_PATH    = PROJECT_ROOT / "models" / "real_dataset.yaml"
RUNS_DIR     = PROJECT_ROOT / "runs"

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.20
TEST_RATIO  = 0.10


def pripremi_dataset():
    """Dijeli stvarne slike na train/val/test."""

    if REAL_DATASET.exists():
        shutil.rmtree(REAL_DATASET)

    slike = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        for slika in REAL_IMAGES.glob(ext):
            label = REAL_LABELS / (slika.stem + ".txt")
            if label.exists():
                slike.append(slika)

    print(f"Pronadjeno {len(slike)} stvarnih slika s anotacijama")

    random.seed(42)
    random.shuffle(slike)

    n_train = int(len(slike) * TRAIN_RATIO)
    n_val   = int(len(slike) * VAL_RATIO)

    train_slike = slike[:n_train]
    val_slike   = slike[n_train:n_train + n_val]
    test_slike  = slike[n_train + n_val:]

    print(f"  Train: {len(train_slike)} | Val: {len(val_slike)} | Test: {len(test_slike)}")

    for split, split_slike in [("train", train_slike),
                                ("val",   val_slike),
                                ("test",  test_slike)]:
        img_dir   = REAL_DATASET / "images" / split
        label_dir = REAL_DATASET / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        for slika in split_slike:
            shutil.copy2(slika, img_dir / slika.name)
            label_src = REAL_LABELS / (slika.stem + ".txt")
            shutil.copy2(label_src, label_dir / (slika.stem + ".txt"))

    print("Dataset podijeljen!")
    return len(train_slike), len(val_slike), len(test_slike)


def napravi_yaml():
    """Kreira dataset.yaml za stvarni dataset."""
    config = {
        "path": str(REAL_DATASET.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    4,
        "names": {
            0: "solid",
            1: "striped",
            2: "white",
            3: "black",
        }
    }
    with open(YAML_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"YAML konfiguracija spremljena: {YAML_PATH}")


def main():
    print("=" * 55)
    print("FINE-TUNING na stvarnim slikama")
    print("=" * 55)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("GRESKA: ultralytics nije instaliran!")
        return

    base_model = RUNS_DIR / "billiard_detector" / "weights" / "best.pt"
    if not base_model.exists():
        print(f"GRESKA: Bazni model nije pronađen: {base_model}")
        print("Pokrenite prvo: python models/train.py")
        return

    print("\n1. Priprema dataseta...")
    n_train, n_val, n_test = pripremi_dataset()
    napravi_yaml()

    print(f"\n2. Pokretanje fine-tuninga...")
    print(f"   Baza: {base_model}")
    print(f"   Train: {n_train} | Val: {n_val} | Test: {n_test}")
    print(f"   Ovo traje ~15-30 minuta...\n")

    model = YOLO(str(base_model))

    results = model.train(
        data       = str(YAML_PATH),
        epochs     = 80,
        imgsz      = 640,
        batch      = 8,
        lr0        = 0.0005,
        lrf        = 0.01,
        patience   = 20,
        project    = str(RUNS_DIR),
        name       = "billiard_real",
        exist_ok   = True,
        plots      = True,
        verbose    = True,
        hsv_h      = 0.015,
        hsv_s      = 0.7,
        hsv_v      = 0.4,
        fliplr     = 0.5,
        degrees    = 10.0,
        translate  = 0.1,
        scale      = 0.3,
        mosaic     = 0.5,
    )

    best = RUNS_DIR / "billiard_real" / "weights" / "best.pt"

    print("\n" + "=" * 55)
    print("FINE-TUNING ZAVRSEN!")
    print(f"Novi model: {best}")

    try:
        print(f"\nFinalne metrike:")
        print(f"  mAP@0.5:    {results.results_dict.get('metrics/mAP50(B)', 0):.3f}")
        print(f"  Preciznost: {results.results_dict.get('metrics/precision(B)', 0):.3f}")
        print(f"  Odziv:      {results.results_dict.get('metrics/recall(B)', 0):.3f}")
    except Exception:
        pass

    print(f"\nSljedeci korak - testiraj novi model:")
    print(f"  python evaluate_real.py --model {best}")
    print("=" * 55)


if __name__ == "__main__":
    main()