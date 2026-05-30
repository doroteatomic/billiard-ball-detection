#!/usr/bin/env python3
import argparse
from pathlib import Path
from collections import Counter
import cv2
from ultralytics import YOLO

MODEL_DEFAULT = Path("runs/billiard_detector/weights/best.pt")
IMAGES_DIR    = Path("real_dataset/images")
RESULTS_DIR   = Path("real_dataset/results")

CLASS_LABELS = {
    "solid":   "Puna",
    "striped": "Sarena",
    "white":   "Bijela",
    "black":   "Crna",
}
CLASS_COLORS = {
    "solid":   (0,   200,  50),
    "striped": (50,  150, 255),
    "white":   (220, 220, 220),
    "black":   (80,   80,  80),
}


def filtriraj_detekcije(detections):
    bijele = [d for d in detections if d[0] == "white"]
    crne   = [d for d in detections if d[0] == "black"]
    ostale = [d for d in detections if d[0] not in ("white", "black")]
    rezultat = []
    if bijele:
        rezultat.append(max(bijele, key=lambda d: d[1]))
    if crne:
        rezultat.append(max(crne, key=lambda d: d[1]))
    rezultat.extend(ostale)
    return rezultat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=str(MODEL_DEFAULT),
                        help="Putanja do modela")
    parser.add_argument("--conf",  type=float, default=0.5,
                        help="Prag pouzdanosti (default: 0.5)")
    args = parser.parse_args()

    out_name = "results_real" if "billiard_real" in args.model else "results_old"
    results_dir = RESULTS_DIR.parent / out_name
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model:  {args.model}")
    print(f"Prag:   {args.conf}")
    print("Ucitavanje modela...")

    model = YOLO(args.model)
    model.conf = args.conf

    slike = list(IMAGES_DIR.glob("*.jpg")) + \
            list(IMAGES_DIR.glob("*.jpeg")) + \
            list(IMAGES_DIR.glob("*.png"))

    if not slike:
        print("Nema slika u real_dataset/images/!")
        return

    print(f"Pronadjeno {len(slike)} slika.\n")

    ukupno = Counter()

    for i, slika_path in enumerate(slike, 1):
        frame = cv2.imread(str(slika_path))
        if frame is None:
            continue

        results = model(frame, verbose=False)[0]
        detections = []
        for box in results.boxes:
            class_id     = int(box.cls[0])
            conf         = float(box.conf[0])
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            name         = model.names[class_id]
            detections.append((name, conf, x1, y1, x2, y2))

        detections = filtriraj_detekcije(detections)
        ukupno += Counter(d[0] for d in detections)

        for name, conf, x1, y1, x2, y2 in detections:
            color = CLASS_COLORS.get(name, (255, 255, 255))
            label = f"{CLASS_LABELS.get(name, name)} {conf:.0%}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imwrite(str(results_dir / f"result_{slika_path.name}"), frame)

        counts = Counter(d[0] for d in detections)
        print(f"  [{i}/{len(slike)}] {slika_path.name[:45]}")
        print(f"         " + " | ".join(
            f"{CLASS_LABELS.get(k,k)}: {v}" for k, v in counts.items()))

    print("\n" + "="*55)
    print(f"GOTOVO! Obradjeno {len(slike)} slika.")
    print(f"Rezultati: {results_dir}")
    print(f"\nUkupno detektirano:")
    for cls in ["solid", "striped", "white", "black"]:
        print(f"  {CLASS_LABELS.get(cls, cls):<10} {ukupno.get(cls, 0):>5}")
    print("="*55)


if __name__ == "__main__":
    main()