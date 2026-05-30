#!/usr/bin/env python3
from pathlib import Path
import shutil

LABELS_DIR  = Path("real_dataset/labels")
LABELS2_DIR = Path("real_dataset/labels2")

REMAP_PRVA = {
    0: 3,
    1: 2,
    2: 0,
    3: 1,
}

REMAP_DRUGA = {
    0: 0,
    1: 1,
    2: 3,
    3: 2,
}


def popravi_folder(folder: Path, remap: dict, naziv: str):
    txt_files = list(folder.glob("*.txt"))
    print(f"\n{naziv}: {len(txt_files)} datoteka")
    for txt_path in txt_files:
        linije = txt_path.read_text().strip().splitlines()
        nove_linije = []
        for linija in linije:
            if not linija.strip():
                continue
            dijelovi = linija.split()
            stari_id = int(dijelovi[0])
            novi_id  = remap.get(stari_id, stari_id)
            dijelovi[0] = str(novi_id)
            nove_linije.append(" ".join(dijelovi))
        txt_path.write_text("\n".join(nove_linije))
    print(f"  Popravljeno!")


def spoji_u_labels(izvor: Path, odrediste: Path):
    kopirano = 0
    for txt_path in izvor.glob("*.txt"):
        dest = odrediste / txt_path.name
        if dest.exists():
            dest = odrediste / f"p2_{txt_path.name}"
        shutil.copy2(txt_path, dest)
        kopirano += 1
    print(f"  Kopirano {kopirano} datoteka u labels/")


def main():
    print("=" * 50)
    print("Popravak klasa anotacija")
    print("=" * 50)

    if not LABELS_DIR.exists():
        print(f"GRESKA: {LABELS_DIR} ne postoji!")
        return
    if not LABELS2_DIR.exists():
        print(f"GRESKA: {LABELS2_DIR} ne postoji!")
        return

    popravi_folder(LABELS_DIR,  REMAP_PRVA,      "Prva (labels/)")
    popravi_folder(LABELS2_DIR, REMAP_DRUGA, "Druga (labels2/)")

    print("\nSpajanje labels2/ u labels/...")
    spoji_u_labels(LABELS2_DIR, LABELS_DIR)

    ukupno = len(list(LABELS_DIR.glob("*.txt")))
    print(f"\n{'='*50}")
    print(f"GOTOVO! Ukupno {ukupno} anotacijskih datoteka u labels/")
    print(f"\nRedoslijed klasa (ispravan):")
    print(f"  0 = solid   (pune kugle)")
    print(f"  1 = striped (sarene kugle)")
    print(f"  2 = white   (bijela kugla)")
    print(f"  3 = black   (crna kugla)")
    print(f"\nSljedeci korak: python retrain_real.py")
    print("=" * 50)


if __name__ == "__main__":
    main()