#!/usr/bin/env python3

import sys
import time
import argparse
import math
from pathlib import Path
from collections import deque, Counter

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_MODEL = PROJECT_ROOT / "runs" / "billiard_detector" / "weights" / "best.pt"

# Klase i boje
CLASS_NAMES = ["solid", "striped", "white", "black"]
CLASS_LABELS_HR = {
    "solid":   "Puna",
    "striped": "Šarena",
    "white":   "Bijela (cue)",
    "black":   "Crna (8)",
}

CLASS_COLORS = {
    "solid":   (  0, 200,  50),   # zelena
    "striped": ( 50, 150, 255),   # narančasta
    "white":   (220, 220, 220),   # bijela
    "black":   ( 80,  80,  80),   # tamno siva
}

CLASS_ICONS = {
    "solid":   "🟢",
    "striped": "🟠",
    "white":   "⚪",
    "black":   "⚫",
}

CONF_THRESHOLD = 0.35
IOU_THRESHOLD  = 0.45


PANEL_W      = 280          
FONT         = cv2.FONT_HERSHEY_SIMPLEX
FONT_SMALL   = 0.45
FONT_MEDIUM  = 0.6
FONT_LARGE   = 0.8
FPS_HISTORY  = 30          

UI_BG        = (20,  22,  28)
UI_ACCENT    = (  0, 200, 120)
UI_TEXT      = (220, 220, 220)
UI_DIM       = (100, 100, 100)
UI_WARNING   = ( 20, 100, 255)



class BilliardDetector:
    """Wrapper oko YOLOv8 modela za detekciju kugli."""

    def __init__(self, model_path: str):
        try:
            from ultralytics import YOLO
        except ImportError:
            print("❌ ultralytics nije instaliran. Pokrenite: pip install ultralytics")
            sys.exit(1)

        model_path = Path(model_path)
        if not model_path.exists():
            print(f"❌ Model nije pronađen: {model_path}")
            print("   Najprije trenirajte model: python models/train.py")
            sys.exit(1)

        print(f"📥 Učitavanje modela: {model_path}")
        self.model = YOLO(str(model_path))
        self.model.conf = CONF_THRESHOLD
        self.model.iou  = IOU_THRESHOLD
        print("✅ Model učitan!")

    def detect(self, frame: np.ndarray) -> list:
       
        results = self.model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            class_id = int(box.cls[0])
            conf      = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            detections.append({
                "class_id":   class_id,
                "class_name": CLASS_NAMES[class_id],
                "conf":       conf,
                "bbox":       (x1, y1, x2, y2),
                "center":     (cx, cy),
            })

        return detections



class Visualizer:

    def __init__(self):
        self.fps_buffer = deque(maxlen=FPS_HISTORY)
        self.frame_count = 0
        self.start_time  = time.time()
        self.detection_history = deque(maxlen=10)

    def update_fps(self, dt: float):
        if dt > 0:
            self.fps_buffer.append(1.0 / dt)

    @property
    def fps(self) -> float:
        return sum(self.fps_buffer) / len(self.fps_buffer) if self.fps_buffer else 0

    def draw_ball_overlay(self, frame: np.ndarray,
                          detections: list) -> np.ndarray:
        overlay = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cx, cy = det["center"]
            name   = det["class_name"]
            conf   = det["conf"]
            color  = CLASS_COLORS[name]
            label  = CLASS_LABELS_HR[name]

            r = max(10, (x2 - x1) // 2)

            cv2.circle(overlay, (cx, cy), r + 3, (0, 0, 0), 2)
            cv2.circle(overlay, (cx, cy), r + 2, color,     2)

            cv2.circle(overlay, (cx, cy), r, color, -1)

            label_text = f"{label} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(label_text, FONT, FONT_SMALL, 1)

            tag_x = cx - tw // 2 - 4
            tag_y = y1 - 8

            cv2.rectangle(overlay,
                          (tag_x, tag_y - th - 6),
                          (tag_x + tw + 8, tag_y + 2),
                          (10, 10, 10), -1)
            cv2.rectangle(overlay,
                          (tag_x, tag_y - th - 6),
                          (tag_x + tw + 8, tag_y + 2),
                          color, 1)

            cv2.putText(overlay, label_text,
                        (tag_x + 4, tag_y - 2),
                        FONT, FONT_SMALL, UI_TEXT, 1, cv2.LINE_AA)

            cv2.circle(overlay, (cx, cy), 3, (255, 255, 255), -1)

        result = cv2.addWeighted(frame, 0.35, overlay, 0.65, 0)
        return result

    def draw_side_panel(self, canvas: np.ndarray,
                        detections: list,
                        frame_w: int) -> None:
        h, w = canvas.shape[:2]
        px = frame_w  # Panel počinje ovdje

        cv2.rectangle(canvas, (px, 0), (w, h), UI_BG, -1)

        y = 20
        padding = 14

        cv2.putText(canvas, "BILJARSKI", (px + padding, y + 14),
                    FONT, FONT_MEDIUM, UI_ACCENT, 1, cv2.LINE_AA)
        y += 22
        cv2.putText(canvas, "DETEKTOR", (px + padding, y + 14),
                    FONT, FONT_MEDIUM, UI_ACCENT, 2, cv2.LINE_AA)
        y += 28

        cv2.line(canvas, (px + padding, y), (w - padding, y), UI_ACCENT, 1)
        y += 16

        fps_color = UI_ACCENT if self.fps >= 20 else UI_WARNING
        cv2.putText(canvas, f"FPS: {self.fps:.1f}",
                    (px + padding, y), FONT, FONT_SMALL, fps_color, 1, cv2.LINE_AA)
        y += 20

        cv2.putText(canvas, f"Kugle: {len(detections)}",
                    (px + padding, y), FONT, FONT_SMALL, UI_TEXT, 1, cv2.LINE_AA)
        y += 26

        cv2.line(canvas, (px + padding, y), (w - padding, y), UI_DIM, 1)
        y += 14

        counts = Counter(d["class_name"] for d in detections)

        cv2.putText(canvas, "Raspored:", (px + padding, y),
                    FONT, FONT_SMALL, UI_DIM, 1, cv2.LINE_AA)
        y += 18

        for cls in CLASS_NAMES:
            count = counts.get(cls, 0)
            label = CLASS_LABELS_HR[cls]
            color = CLASS_COLORS[cls]

            # Kvadratić s bojom klase
            sq = 10
            cv2.rectangle(canvas,
                          (px + padding, y - sq + 2),
                          (px + padding + sq, y + 2),
                          color, -1)

            text = f"{label}: {count}"
            cv2.putText(canvas, text,
                        (px + padding + sq + 6, y),
                        FONT, FONT_SMALL,
                        UI_TEXT if count > 0 else UI_DIM,
                        1, cv2.LINE_AA)
            y += 18

        y += 6
        cv2.line(canvas, (px + padding, y), (w - padding, y), UI_DIM, 1)
        y += 14

        cv2.putText(canvas, "Detalji:", (px + padding, y),
                    FONT, FONT_SMALL, UI_DIM, 1, cv2.LINE_AA)
        y += 18

        max_show = min(len(detections), (h - y - 60) // 22)
        for i, det in enumerate(detections[:max_show]):
            name  = det["class_name"]
            conf  = det["conf"]
            cx, cy = det["center"]
            color = CLASS_COLORS[name]
            label = CLASS_LABELS_HR[name]

            line = f"{i+1}. {label[:8]} ({conf:.0%})"
            cv2.putText(canvas, line,
                        (px + padding, y),
                        FONT, FONT_SMALL, color, 1, cv2.LINE_AA)

            pos_text = f"   [{cx},{cy}]"
            cv2.putText(canvas, pos_text,
                        (px + padding, y + 13),
                        FONT, 0.35, UI_DIM, 1, cv2.LINE_AA)
            y += 28

        y = h - 55
        cv2.line(canvas, (px + padding, y), (w - padding, y), UI_DIM, 1)
        y += 14
        for tip in ["[Q] Izlaz", "[S] Screenshot", "[P] Pauza"]:
            cv2.putText(canvas, tip, (px + padding, y),
                        FONT, 0.38, UI_DIM, 1, cv2.LINE_AA)
            y += 14

    def draw_table_map(self, canvas: np.ndarray,
                       detections: list,
                       src_w: int, src_h: int,
                       dst_x: int, dst_y: int,
                       map_w: int, map_h: int) -> None:
        cv2.rectangle(canvas, (dst_x, dst_y),
                      (dst_x + map_w, dst_y + map_h),
                      (30, 60, 30), -1)
        cv2.rectangle(canvas, (dst_x, dst_y),
                      (dst_x + map_w, dst_y + map_h),
                      UI_ACCENT, 1)

        label = "Pozicijska mapa"
        cv2.putText(canvas, label,
                    (dst_x + 4, dst_y - 5),
                    FONT, 0.38, UI_DIM, 1, cv2.LINE_AA)

        for det in detections:
            cx, cy = det["center"]
            mx = int(dst_x + (cx / src_w) * map_w)
            my = int(dst_y + (cy / src_h) * map_h)
            color = CLASS_COLORS[det["class_name"]]
            cv2.circle(canvas, (mx, my), 5, color, -1)
            cv2.circle(canvas, (mx, my), 5, (255, 255, 255), 1)

    def compose(self, frame: np.ndarray,
                detections: list) -> np.ndarray:
        fh, fw = frame.shape[:2]

        annotated = self.draw_ball_overlay(frame, detections)

        canvas_w = fw + PANEL_W
        canvas = np.zeros((fh, canvas_w, 3), dtype=np.uint8)
        canvas[:fh, :fw] = annotated

        map_h = min(140, fh // 3)
        map_w = PANEL_W - 28
        self.draw_table_map(canvas, detections,
                            fw, fh,
                            fw + 14, fh - map_h - 60,
                            map_w, map_h)

        self.draw_side_panel(canvas, detections, fw)

        return canvas



def run_video(detector: BilliardDetector, source, output_path=None):
    """Pokretanje na video zapisu ili webcamu."""
    viz = Visualizer()

    if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
        cap = cv2.VideoCapture(int(source))
        print(f"📷 Webcam {source}")
    else:
        cap = cv2.VideoCapture(str(source))
        print(f"🎬 Video: {source}")

    if not cap.isOpened():
        print("❌ Nije moguće otvoriti izvor!")
        return

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"📐 Rezolucija: {fw}×{fh}")

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, 30, (fw + PANEL_W, fh))
        print(f"💾 Snimanje u: {output_path}")

    paused     = False
    screenshot = 0
    prev_time  = time.time()

    print("\n🎱 Detekcija pokrenuta! [Q]=Izlaz [S]=Screenshot [P]=Pauza\n")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("⏹️  Kraj videa.")
                break

            detections = detector.detect(frame)

            now = time.time()
            viz.update_fps(now - prev_time)
            prev_time = now

            canvas = viz.compose(frame, detections)
            viz.frame_count += 1

            if viz.frame_count % 30 == 0:
                counts = Counter(d["class_name"] for d in detections)
                print(f"  Frame {viz.frame_count:5d} | "
                      f"FPS: {viz.fps:5.1f} | "
                      f"Kugle: {len(detections):2d} | "
                      + " ".join(f"{k}:{v}" for k, v in counts.items()))

            if writer:
                writer.write(canvas)

        cv2.imshow("Biljarski Detektor", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("\n👋 Izlaz...")
            break
        elif key == ord("p"):
            paused = not paused
            print("⏸️  Pauza" if paused else "▶️  Nastavak")
        elif key == ord("s"):
            screenshot += 1
            path = PROJECT_ROOT / f"screenshot_{screenshot:03d}.jpg"
            cv2.imwrite(str(path), canvas)
            print(f"📸 Screenshot: {path}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Obrađeno {viz.frame_count} slika")


def run_image(detector: BilliardDetector, image_path: str):
    viz = Visualizer()

    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"❌ Nije moguće učitati sliku: {image_path}")
        return

    print(f"🖼️  Slika: {image_path} | {frame.shape[1]}×{frame.shape[0]}")

    t0 = time.time()
    detections = detector.detect(frame)
    dt = time.time() - t0

    canvas = viz.compose(frame, detections)

    print(f"\n📊 Rezultati ({dt*1000:.1f}ms):")
    print(f"   Detektirano kugli: {len(detections)}")
    counts = Counter(d["class_name"] for d in detections)
    for cls, cnt in counts.items():
        print(f"   {CLASS_LABELS_HR[cls]}: {cnt}")

    out_path = Path(image_path).stem + "_detected.jpg"
    cv2.imwrite(out_path, canvas)
    print(f"\n💾 Rezultat spremljen: {out_path}")

    cv2.imshow("Detekcija", canvas)
    print("Pritisni bilo koju tipku za zatvaranje...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def parse_args():
    parser = argparse.ArgumentParser(
        description="Detektor biljarskih kugli",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Primjeri:
  python app/detector_app.py                       # Webcam
  python app/detector_app.py --source 1            # Drugi webcam
  python app/detector_app.py --source video.mp4    # Video
  python app/detector_app.py --source slika.jpg --mode image
  python app/detector_app.py --output output.mp4   # Snimanje
        """
    )
    parser.add_argument("--source",  default="0",
                        help="Izvor: 0=webcam, putanja do videa/slike")
    parser.add_argument("--model",   default=str(DEFAULT_MODEL),
                        help="Putanja do modela (.pt)")
    parser.add_argument("--mode",    choices=["video", "image"], default="video",
                        help="Mod rada")
    parser.add_argument("--output",  default=None,
                        help="Putanja za snimanje rezultata")
    parser.add_argument("--conf",    type=float, default=CONF_THRESHOLD,
                        help="Prag pouzdanosti detekcije")
    return parser.parse_args()


def main():
    args = parse_args()

    print("🎱 Biljarski Detektor")
    print("=" * 40)

    detector = BilliardDetector(args.model)
    detector.model.conf = args.conf

    if args.mode == "image":
        run_image(detector, args.source)
    else:
        run_video(detector, args.source, args.output)


if __name__ == "__main__":
    main()