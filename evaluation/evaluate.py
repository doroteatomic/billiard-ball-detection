#!/usr/bin/env python3
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH   = PROJECT_ROOT / "runs" / "billiard_detector" / "weights" / "best.pt"
DATASET_YAML = PROJECT_ROOT / "models" / "dataset.yaml"
EVAL_DIR     = PROJECT_ROOT / "evaluation" / "results"

CLASS_NAMES  = ["solid", "striped", "white", "black"]
CLASS_LABELS_HR = ["Pune", "Šarene", "Bijela", "Crna"]
CLASS_COLORS = ["#00C878", "#FF7832", "#DCDCDC", "#505050"]



def run_yolo_eval(model_path: str, conf: float = 0.25) -> dict:
    """Pokreni YOLO evaluaciju na test skupu."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ ultralytics nije instaliran!")
        sys.exit(1)

    print(f"🔍 Evaluacija modela: {model_path}")
    model = YOLO(str(model_path))

    results = model.val(
        data=str(DATASET_YAML),
        split="test",
        imgsz=640,
        conf=conf,
        iou=0.5,
        verbose=True,
        plots=True,
        save_json=True,
    )

    metrics = {
        "map50":    float(results.box.map50),
        "map50_95": float(results.box.map),
        "classes":  []
    }

    for i, cls in enumerate(CLASS_NAMES):
        try:
            metrics["classes"].append({
                "name":  cls,
                "label": CLASS_LABELS_HR[i],
                "ap50":  float(results.box.ap50[i]) if i < len(results.box.ap50) else 0,
                "p":     float(results.box.p[i])    if i < len(results.box.p)    else 0,
                "r":     float(results.box.r[i])    if i < len(results.box.r)    else 0,
            })
        except (IndexError, AttributeError):
            metrics["classes"].append({
                "name": cls, "label": CLASS_LABELS_HR[i],
                "ap50": 0.0, "p": 0.0, "r": 0.0
            })

    return metrics


def benchmark_speed(model_path: str, n_runs: int = 100) -> dict:
    from ultralytics import YOLO
    import torch

    model = YOLO(str(model_path))
    dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # Zagrijavanje
    for _ in range(5):
        model(dummy, verbose=False)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model(dummy, verbose=False)
        times.append(time.perf_counter() - t0)

    return {
        "mean_ms":  np.mean(times) * 1000,
        "std_ms":   np.std(times)  * 1000,
        "min_ms":   np.min(times)  * 1000,
        "max_ms":   np.max(times)  * 1000,
        "fps":      1.0 / np.mean(times),
        "n_runs":   n_runs,
    }



def plot_metrics(metrics: dict, speed: dict, output_dir: Path) -> None:
    """Generira sve grafove evaluacije."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 12), facecolor="#0E1117")
    gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    text_color   = "#E0E0E0"
    grid_color   = "#2A2A3A"
    accent_color = "#00C878"

    def style_ax(ax, title):
        ax.set_facecolor("#161B22")
        ax.tick_params(colors=text_color, labelsize=9)
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)
        ax.set_title(title, color=accent_color, fontsize=11, pad=10, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_color(grid_color)
        ax.grid(color=grid_color, linestyle="--", alpha=0.5)

    classes  = [c["label"] for c in metrics["classes"]]
    ap_vals  = [c["ap50"]  for c in metrics["classes"]]
    p_vals   = [c["p"]     for c in metrics["classes"]]
    r_vals   = [c["r"]     for c in metrics["classes"]]
    colors   = CLASS_COLORS

    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(classes, ap_vals, color=colors, edgecolor="#222", linewidth=0.8)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("AP@0.5", color=text_color)
    for bar, val in zip(bars, ap_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                 f"{val:.3f}", ha="center", color=text_color, fontsize=9)
    style_ax(ax1, "AP@0.5 po klasama")

    ax2 = fig.add_subplot(gs[0, 1])
    x   = np.arange(len(classes))
    w   = 0.35
    ax2.bar(x - w/2, p_vals, w, label="Preciznost", color=accent_color, alpha=0.85)
    ax2.bar(x + w/2, r_vals, w, label="Odziv",      color="#5078FF",    alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes, color=text_color)
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel("Vrijednost", color=text_color)
    ax2.legend(facecolor="#161B22", labelcolor=text_color, fontsize=9)
    style_ax(ax2, "Preciznost i Odziv")

    ax3 = fig.add_subplot(gs[0, 2], projection="polar")
    ax3.set_facecolor("#161B22")

    metric_labels = ["mAP@0.5", "mAP@.5:.95", "Prec.", "Odziv", "F1"]
    avg_p = np.mean(p_vals)
    avg_r = np.mean(r_vals)
    f1    = 2 * avg_p * avg_r / (avg_p + avg_r + 1e-8)
    values = [metrics["map50"], metrics["map50_95"], avg_p, avg_r, f1]
    values_c = values + [values[0]]   

    angles = np.linspace(0, 2 * np.pi, len(metric_labels), endpoint=False).tolist()
    angles += angles[:1]

    ax3.plot(angles, values_c, color=accent_color, linewidth=2)
    ax3.fill(angles, values_c, color=accent_color, alpha=0.25)
    ax3.set_thetagrids(np.degrees(angles[:-1]), metric_labels,
                       color=text_color, fontsize=9)
    ax3.set_ylim(0, 1)
    ax3.grid(color=grid_color, linestyle="--", alpha=0.6)
    ax3.set_title("Ukupne metrike", color=accent_color,
                  fontsize=11, pad=20, fontweight="bold")

    ax4 = fig.add_subplot(gs[1, 0])
    fps_vals = [1000 / speed["mean_ms"]]   
    labels_sp = ["Prosjek FPS"]
    ax4.bar(labels_sp, fps_vals, color=["#FF7832"])
    ax4.axhline(y=30, color="#FF4444", linestyle="--", alpha=0.7, label="30 FPS target")
    ax4.set_ylabel("FPS", color=text_color)
    ax4.set_ylim(0, max(fps_vals) * 1.3)
    ax4.text(0, fps_vals[0] + 1, f"{fps_vals[0]:.1f}", ha="center",
             color=text_color, fontsize=11, fontweight="bold")
    ax4.legend(facecolor="#161B22", labelcolor=text_color, fontsize=9)
    style_ax(ax4, "Brzina zaključivanja")

    ax5 = fig.add_subplot(gs[1, 1])
    lat_labels = ["Min", "Prosjek", "Max"]
    lat_vals   = [speed["min_ms"], speed["mean_ms"], speed["max_ms"]]
    lat_colors = [accent_color, "#FF7832", "#FF4444"]
    ax5.bar(lat_labels, lat_vals, color=lat_colors)
    for i, val in enumerate(lat_vals):
        ax5.text(i, val + 0.3, f"{val:.1f}ms", ha="center",
                 color=text_color, fontsize=9)
    ax5.set_ylabel("ms", color=text_color)
    style_ax(ax5, "Latencija zaključivanja")

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    ax6.set_facecolor("#161B22")

    summary_data = [
        ["Metrika", "Vrijednost"],
        ["mAP@0.5",      f"{metrics['map50']:.4f}"],
        ["mAP@0.5:0.95", f"{metrics['map50_95']:.4f}"],
        ["Prec. (avg)",  f"{avg_p:.4f}"],
        ["Odziv (avg)",  f"{avg_r:.4f}"],
        ["F1 Score",     f"{f1:.4f}"],
        ["FPS",          f"{speed['fps']:.1f}"],
        ["Lat. (avg)",   f"{speed['mean_ms']:.1f} ms"],
    ]

    table = ax6.table(
        cellText=summary_data[1:],
        colLabels=summary_data[0],
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    for (row, col), cell in table.get_celld().items():
        cell.set_facecolor("#1E2530" if row % 2 == 0 else "#161B22")
        cell.set_text_props(color=text_color)
        cell.set_edgecolor(grid_color)
        if row == 0:
            cell.set_facecolor("#0D2B1A")
            cell.set_text_props(color=accent_color, fontweight="bold")

    ax6.set_title("Sažetak evaluacije", color=accent_color,
                  fontsize=11, pad=10, fontweight="bold")

    # Naslov
    fig.suptitle("🎱 Evaluacija Detektora Biljarskih Kugli",
                 color=text_color, fontsize=15, fontweight="bold", y=0.98)

    # Spremi
    out_path = output_dir / "evaluation_report.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"📊 Grafovi spremljeni: {out_path}")


def print_report(metrics: dict, speed: dict) -> None:
    """Ispisuje izvještaj u terminal."""
    print("\n" + "=" * 55)
    print("🎱  IZVJEŠTAJ EVALUACIJE – BILJARSKI DETEKTOR")
    print("=" * 55)

    print(f"\n📈 Ukupne metrike:")
    print(f"   mAP@0.5:       {metrics['map50']:.4f}")
    print(f"   mAP@0.5:0.95:  {metrics['map50_95']:.4f}")

    print(f"\n📋 Metrike po klasama:")
    print(f"   {'Klasa':<12} {'AP@0.5':>8} {'Preciznost':>12} {'Odziv':>8}")
    print(f"   {'-'*44}")
    for cls in metrics["classes"]:
        print(f"   {cls['label']:<12} {cls['ap50']:>8.4f} {cls['p']:>12.4f} {cls['r']:>8.4f}")

    print(f"\n⚡ Brzina zaključivanja ({speed['n_runs']} pokretanja):")
    print(f"   FPS:           {speed['fps']:.1f}")
    print(f"   Latencija:     {speed['mean_ms']:.1f} ± {speed['std_ms']:.1f} ms")
    print(f"   Min/Max:       {speed['min_ms']:.1f} / {speed['max_ms']:.1f} ms")

    avg_p = np.mean([c["p"] for c in metrics["classes"]])
    avg_r = np.mean([c["r"] for c in metrics["classes"]])
    f1    = 2 * avg_p * avg_r / (avg_p + avg_r + 1e-8)
    print(f"\n🏆 F1 Score (avg): {f1:.4f}")

    map50 = metrics["map50"]
    if   map50 >= 0.90: grade = "Odlično ⭐⭐⭐"
    elif map50 >= 0.75: grade = "Vrlo dobro ⭐⭐"
    elif map50 >= 0.60: grade = "Dobro ⭐"
    elif map50 >= 0.45: grade = "Prihvatljivo ⚠️"
    else:               grade = "Treba poboljšanje ❌"

    print(f"\n🎯 Ocjena: {grade}")
    print("=" * 55)


def save_report_json(metrics: dict, speed: dict, output_dir: Path) -> None:
    report = {"metrics": metrics, "speed": speed}
    path   = output_dir / "evaluation_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"📄 JSON izvještaj: {path}")



def main():
    parser = argparse.ArgumentParser(description="Evaluacija detektora biljarskih kugli")
    parser.add_argument("--model",      default=str(MODEL_PATH),
                        help="Putanja do modela (.pt)")
    parser.add_argument("--conf",       type=float, default=0.25,
                        help="Prag pouzdanosti")
    parser.add_argument("--speed-runs", type=int,   default=50,
                        help="Broj pokretanja za benchmark brzine")
    parser.add_argument("--no-plots",   action="store_true",
                        help="Preskoči generiranje grafova")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model nije pronađen: {model_path}")
        print("   Pokrenite: python models/train.py")
        sys.exit(1)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    print("🎱 Evaluacija detektora biljarskih kugli")
    print("=" * 40)

    print("\n1️⃣  Evaluacija točnosti detekcije...")
    metrics = run_yolo_eval(str(model_path), args.conf)

    print(f"\n2️⃣  Benchmark brzine ({args.speed_runs} pokretanja)...")
    speed = benchmark_speed(str(model_path), args.speed_runs)

    print_report(metrics, speed)
    save_report_json(metrics, speed, EVAL_DIR)

    if not args.no_plots:
        print("\n3️⃣  Generiranje grafova...")
        plot_metrics(metrics, speed, EVAL_DIR)

    print(f"\n✅ Evaluacija završena! Rezultati: {EVAL_DIR}")


if __name__ == "__main__":
    main()