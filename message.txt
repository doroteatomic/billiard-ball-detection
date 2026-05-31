#!/usr/bin/env python3
"""
Streamlit web aplikacija za detekciju biljarskih kugli.
Pokretanje: streamlit run streamlit_app.py
"""

import io
import base64
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

# ─── Konfiguracija ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Detekcija Biljarskih Kugli",
    page_icon="🎱",
    layout="wide",
)

CLASS_LABELS = {
    "solid":   "Puna",
    "striped": "Šarena",
    "white":   "Bijela (cue)",
    "black":   "Crna (8)",
}
CLASS_COLORS_BGR = {
    "solid":   (0,   200,  50),
    "striped": (50,  150, 255),
    "white":   (220, 220, 220),
    "black":   (80,   80,  80),
}
CLASS_ICONS = {
    "solid":   "🟢",
    "striped": "🟠",
    "white":   "⚪",
    "black":   "⚫",
}

MODEL_PATHS = [
    "runs/detect/runs/billiard_real/weights/best.pt",
    "runs/billiard_real/weights/best.pt",
    "runs/billiard_detector/weights/best.pt",
]


# ─── Učitavanje modela ────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    for path in MODEL_PATHS:
        if Path(path).exists():
            return YOLO(path), path
    return None, None


# ─── Detekcija ────────────────────────────────────────────────────────────────

def detect(frame: np.ndarray, model, conf: float):
    model.conf = conf
    results = model(frame, verbose=False)[0]
    detections = []

    for box in results.boxes:
        class_id     = int(box.cls[0])
        conf_val     = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        name         = model.names[class_id]
        detections.append({
            "name":  name,
            "label": CLASS_LABELS.get(name, name),
            "conf":  conf_val,
            "bbox":  (x1, y1, x2, y2),
            "center": ((x1 + x2) // 2, (y1 + y2) // 2),
        })

    return detections


def draw_detections(frame: np.ndarray, detections: list) -> np.ndarray:
    result = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = CLASS_COLORS_BGR.get(det["name"], (255, 255, 255))
        label = f"{det['label']} {det['conf']:.0%}"

        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(result, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(result, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    return result


# ─── UI ───────────────────────────────────────────────────────────────────────

def main():
    # Naslov
    st.title("🎱 Detekcija Biljarskih Kugli")
    st.markdown("YOLOv8 model · Klasifikacija: **Puna · Šarena · Bijela · Crna**")
    st.divider()

    # Učitaj model
    model, model_path = load_model()
    if model is None:
        st.error("Model nije pronađen! Pokrenite treniranje: `python models/train.py`")
        return

    st.success(f"Model učitan: `{model_path}`")

    # Sidebar – postavke
    with st.sidebar:
        st.header("Postavke")
        conf = st.slider("Prag pouzdanosti", 0.1, 0.95, 0.5, 0.05,
                         help="Viši prag = manje ali točnije detekcije")
        st.divider()
        st.markdown("### Klase")
        for name, label in CLASS_LABELS.items():
            icon = CLASS_ICONS[name]
            st.markdown(f"{icon} **{label}**")
        st.divider()
        st.markdown("### O projektu")
        st.markdown("Projektni zadatak iz **Strojnog učenja**")
        st.markdown("Dataset: sintetički + stvarne slike")

    # Glavni dio – upload
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Učitaj sliku")
        uploaded = st.file_uploader(
            "Odaberi sliku biljarskog stola",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

    if uploaded is not None:
        # Učitaj sliku
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with col1:
            st.image(frame_rgb, caption="Originalna slika", use_container_width=True)

        # Detekcija
        with st.spinner("Detektiram kugle..."):
            detections = detect(frame, model, conf)
            result_frame = draw_detections(frame, detections)
            result_rgb   = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)

        with col2:
            st.subheader("Rezultat detekcije")
            st.image(result_rgb, caption=f"Detektirano kugli: {len(detections)}",
                     use_container_width=True)

        st.divider()

        # Statistike
        counts = Counter(d["name"] for d in detections)

        st.subheader("Statistike")
        cols = st.columns(4)
        for i, (name, label) in enumerate(CLASS_LABELS.items()):
            with cols[i]:
                icon  = CLASS_ICONS[name]
                count = counts.get(name, 0)
                st.metric(f"{icon} {label}", count)

        st.divider()

        # Detalji detekcija
        if detections:
            st.subheader("Detalji")
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("**Lista detektiranih kugli:**")
                for i, det in enumerate(detections, 1):
                    icon = CLASS_ICONS.get(det["name"], "🔵")
                    cx, cy = det["center"]
                    st.markdown(
                        f"{i}. {icon} **{det['label']}** — "
                        f"pouzdanost: `{det['conf']:.1%}` — "
                        f"pozicija: `({cx}, {cy})`"
                    )

            with col_b:
                st.markdown("**Pozicijska mapa:**")
                h, w = frame.shape[:2]
                map_img = np.zeros((200, 300, 3), dtype=np.uint8)
                map_img[:] = (30, 80, 30)
                cv2.rectangle(map_img, (0, 0), (299, 199), (0, 180, 60), 2)

                for det in detections:
                    cx, cy = det["center"]
                    mx = int(cx / w * 280) + 10
                    my = int(cy / h * 180) + 10
                    color = CLASS_COLORS_BGR.get(det["name"], (255, 255, 255))
                    cv2.circle(map_img, (mx, my), 7, color, -1)
                    cv2.circle(map_img, (mx, my), 7, (255, 255, 255), 1)

                map_rgb = cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB)
                st.image(map_rgb, caption="Pozicije kugli na stolu",
                         use_container_width=True)

            # Download rezultata
            st.divider()
            _, buf = cv2.imencode(".jpg", result_frame,
                                  [cv2.IMWRITE_JPEG_QUALITY, 92])
            st.download_button(
                "Preuzmi označenu sliku",
                data=buf.tobytes(),
                file_name="detekcija_rezultat.jpg",
                mime="image/jpeg",
            )

    else:
        with col2:
            st.info("Učitaj sliku s lijeve strane za početak detekcije")


if __name__ == "__main__":
    main()