#!/usr/bin/env python3
"""
Generiranje sintetičkog dataseta biljarskih kugli.
Stvara realistične slike s označenim kuglama u YOLO formatu.
"""

import os
import json
import random
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from pathlib import Path
from tqdm import tqdm


OUTPUT_DIR = Path(__file__).parent.parent / "dataset"
TRAIN_COUNT = 1800
VAL_COUNT   = 360
TEST_COUNT  = 180

IMG_W, IMG_H = 640, 640
BALL_RADIUS_MIN, BALL_RADIUS_MAX = 22, 38

CLASS_NAMES = ["solid", "striped", "white", "black"]

SOLID_COLORS = [
    (230, 210,  30),   # 1 – žuta
    (  0,  80, 200),   # 2 – plava
    (210,  30,  30),   # 3 – crvena
    (130,   0, 160),   # 4 – ljubičasta
    (230, 110,  10),   # 5 – narančasta
    ( 10, 140,  10),   # 6 – zelena
    (140,  20,  20),   # 7 – bordo
]

STRIPED_COLORS = SOLID_COLORS  

# Boje stola
TABLE_COLORS = [
    ( 20, 110,  40),   # klasična zelena
    ( 15,  80, 120),   # plava
    ( 60,  30,  10),   # smeđa
    ( 15,  95,  55),   # tamno zelena
]



def random_table_bg(w: int, h: int) -> Image.Image:
    """Generira pozadinu stola s teksturom tkanine."""
    base = random.choice(TABLE_COLORS)
    # Dodaj varijaciju osvjetljenja
    brightness = random.uniform(0.7, 1.3)
    base = tuple(min(255, int(c * brightness)) for c in base)

    img = Image.new("RGB", (w, h), base)
    draw = ImageDraw.Draw(img)

    pixels = np.array(img, dtype=np.int16)
    noise = np.random.randint(-12, 12, pixels.shape, dtype=np.int16)
    pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(pixels)

    vignette = Image.new("L", (w, h), 0)
    vd = ImageDraw.Draw(vignette)
    for i in range(min(w, h) // 4):
        alpha = int(60 * (i / (min(w, h) // 4)))
        vd.rectangle([i, i, w - i, h - i], outline=alpha)
    img = Image.composite(img, Image.new("RGB", (w, h), (0, 0, 0)),
                          Image.fromarray(255 - np.array(vignette)))

    return img


def draw_solid_ball(draw: ImageDraw.Draw, cx: int, cy: int, r: int,
                    color: tuple) -> None:
    shadow_offset = max(2, r // 5)
    draw.ellipse([cx - r + shadow_offset, cy - r + shadow_offset,
                  cx + r + shadow_offset, cy + r + shadow_offset],
                 fill=(0, 0, 0, 80))

    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)

    hr = max(2, r // 3)
    hx = cx - r // 3
    hy = cy - r // 3
    draw.ellipse([hx - hr // 2, hy - hr // 3,
                  hx + hr // 2, hy + hr // 3],
                 fill=(255, 255, 255, 180))

    nr = max(3, r // 3)
    draw.ellipse([cx - nr, cy - nr, cx + nr, cy + nr],
                 fill=(255, 255, 255))
    draw.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=(50, 50, 50))


def draw_striped_ball(draw: ImageDraw.Draw, cx: int, cy: int, r: int,
                      color: tuple) -> None:
    """Crta šarenu (prugastu) biljarsku kuglu."""
    shadow_offset = max(2, r // 5)
    draw.ellipse([cx - r + shadow_offset, cy - r + shadow_offset,
                  cx + r + shadow_offset, cy + r + shadow_offset],
                 fill=(0, 0, 0, 80))

    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(240, 240, 240))

    stripe_h = int(r * 0.55)
    draw.rectangle([cx - r, cy - stripe_h, cx + r, cy + stripe_h], fill=color)

    nr = max(3, r // 3)
    draw.ellipse([cx - nr, cy - nr, cx + nr, cy + nr], fill=(255, 255, 255))
    draw.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=(50, 50, 50))

    hr = max(2, r // 3)
    hx = cx - r // 3
    hy = cy - r // 3
    draw.ellipse([hx - hr // 2, hy - hr // 3,
                  hx + hr // 2, hy + hr // 3],
                 fill=(255, 255, 255, 160))

    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(220, 220, 220), width=1)


def draw_white_ball(draw: ImageDraw.Draw, cx: int, cy: int, r: int) -> None:
    shadow_offset = max(2, r // 5)
    draw.ellipse([cx - r + shadow_offset, cy - r + shadow_offset,
                  cx + r + shadow_offset, cy + r + shadow_offset],
                 fill=(0, 0, 0, 60))
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(245, 245, 245))
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(200, 200, 200), width=1)
    hr = max(3, r // 3)
    hx = cx - r // 4
    hy = cy - r // 4
    draw.ellipse([hx - hr // 2, hy - hr // 3,
                  hx + hr // 2, hy + hr // 3],
                 fill=(255, 255, 255, 230))


def draw_black_ball(draw: ImageDraw.Draw, cx: int, cy: int, r: int) -> None:
    shadow_offset = max(2, r // 5)
    draw.ellipse([cx - r + shadow_offset, cy - r + shadow_offset,
                  cx + r + shadow_offset, cy + r + shadow_offset],
                 fill=(0, 0, 0, 100))
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(20, 20, 20))
    nr = max(3, r // 3)
    draw.ellipse([cx - nr, cy - nr, cx + nr, cy + nr], fill=(255, 255, 255))
    draw.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=(20, 20, 20))
    hr = max(2, r // 4)
    hx = cx - r // 3
    hy = cy - r // 3
    draw.ellipse([hx - hr // 2, hy - hr // 3,
                  hx + hr // 2, hy + hr // 3],
                 fill=(100, 100, 100, 180))


def place_balls(n_balls: int, w: int, h: int,
                r_min: int, r_max: int) -> list:
    balls = []
    margin = r_max + 5
    attempts = 0
    while len(balls) < n_balls and attempts < 5000:
        attempts += 1
        r  = random.randint(r_min, r_max)
        cx = random.randint(margin, w - margin)
        cy = random.randint(margin, h - margin)
        overlap = False
        for (ox, oy, or_, _) in balls:
            dist = math.sqrt((cx - ox) ** 2 + (cy - oy) ** 2)
            if dist < (r + or_) * 0.85:   
                overlap = True
                break
        if not overlap:
            balls.append((cx, cy, r, None))  
    return balls


def generate_image(split: str) -> tuple:
    w, h = IMG_W, IMG_H

    img = random_table_bg(w, h)
    draw = ImageDraw.Draw(img, "RGBA")

    n_balls = random.randint(3, 12)
    positions = place_balls(n_balls, w, h, BALL_RADIUS_MIN, BALL_RADIUS_MAX)

    annotations = []

    for cx, cy, r, _ in positions:
        ball_type = random.choices(
            ["solid", "striped", "white", "black"],
            weights=[40, 40, 15, 5]
        )[0]

        if ball_type == "solid":
            color = random.choice(SOLID_COLORS)
            color = tuple(min(255, max(0, c + random.randint(-20, 20))) for c in color)
            draw_solid_ball(draw, cx, cy, r, color)
            class_id = 0

        elif ball_type == "striped":
            color = random.choice(STRIPED_COLORS)
            color = tuple(min(255, max(0, c + random.randint(-20, 20))) for c in color)
            draw_striped_ball(draw, cx, cy, r, color)
            class_id = 1

        elif ball_type == "white":
            draw_white_ball(draw, cx, cy, r)
            class_id = 2

        else:  # black
            draw_black_ball(draw, cx, cy, r)
            class_id = 3

        bw = (2 * r) / w
        bh = (2 * r) / h
        annotations.append([
            class_id,
            cx / w,
            cy / h,
            bw,
            bh
        ])

    img = img.convert("RGB")

    if random.random() < 0.5:
        factor = random.uniform(0.5, 1.6)
        img = ImageEnhance.Brightness(img).enhance(factor)

    if random.random() < 0.4:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.4))

    if random.random() < 0.2:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

    if random.random() < 0.3:
        arr = np.array(img, dtype=np.int16)
        noise = np.random.randint(-20, 20, arr.shape, dtype=np.int16)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        annotations = [[c, 1 - cx, cy, bw, bh]
                        for c, cx, cy, bw, bh in annotations]

    return img, annotations



def generate_split(split: str, count: int) -> None:
    img_dir   = OUTPUT_DIR / "images" / split
    label_dir = OUTPUT_DIR / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📸 Generiranje {count} slika za split '{split}'...")
    for i in tqdm(range(count), desc=split):
        img, annotations = generate_image(split)

        # Spremi sliku
        img_path = img_dir / f"{split}_{i:05d}.jpg"
        img.save(img_path, "JPEG", quality=92)

        label_path = label_dir / f"{split}_{i:05d}.txt"
        with open(label_path, "w") as f:
            for ann in annotations:
                f.write(" ".join(f"{v:.6f}" for v in ann) + "\n")


def create_dataset_yaml() -> None:
    yaml_content = f"""# Dataset konfiguracija za detekciju biljarskih kugli
path: {OUTPUT_DIR.resolve()}
train: images/train
val: images/val
test: images/test

nc: 4
names:
  0: solid
  1: striped
  2: white
  3: black
"""
    yaml_path = Path(__file__).parent.parent / "models" / "dataset.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"\n✅ Dataset YAML spremljen: {yaml_path}")


def main():
    print("🎱 Generator sintetičkog dataseta biljarskih kugli")
    print("=" * 50)

    random.seed(42)
    np.random.seed(42)

    generate_split("train", TRAIN_COUNT)
    generate_split("val",   VAL_COUNT)
    generate_split("test",  TEST_COUNT)

    create_dataset_yaml()

    total = TRAIN_COUNT + VAL_COUNT + TEST_COUNT
    print(f"\n✅ Dataset generiran! Ukupno {total} slika.")
    print(f"   📁 Lokacija: {OUTPUT_DIR}")
    print(f"   🏋️  Train: {TRAIN_COUNT} | 🔍 Val: {VAL_COUNT} | 🧪 Test: {TEST_COUNT}")
    print("\nSlijedeći korak: python models/train.py")


if __name__ == "__main__":
    main()