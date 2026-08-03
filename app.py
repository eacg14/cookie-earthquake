"""Streamlit entrypoint: upload a photo series, pick a platform/format, review
and correct the auto-detected alignment, then export cropped/aligned stills.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageOps
from streamlit_image_coordinates import streamlit_image_coordinates

from flipbook import export, pipeline
from flipbook.detection import PoseDetector, alignment_record_from_landmarks
from flipbook.models import AlignmentRecord
from flipbook.presets import PRESETS

MODEL_PATH = Path(__file__).resolve().parent / "models" / "pose_landmarker_full.task"
THUMBNAIL_WIDTH = 420
PREVIEW_WIDTH = 320

st.set_page_config(page_title="Flipbook Aligner", layout="wide")


@st.cache_resource
def get_detector(model_path: str) -> PoseDetector:
    return PoseDetector(model_path)


def load_frame(uploaded_file) -> np.ndarray:
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image)
    return np.array(image.convert("RGB"))


def file_signature(uploaded_files) -> str:
    hasher = hashlib.sha256()
    for f in uploaded_files:
        hasher.update(f.name.encode())
        hasher.update(str(f.size).encode())
    return hasher.hexdigest()


def init_state() -> None:
    defaults = {
        "upload_signature": None,
        "frames": [],
        "frame_names": [],
        "auto_records": {},  # idx -> (PoseResult, AlignmentRecord, warnings)
        "manual_records": {},  # idx -> AlignmentRecord
        "reference_frame_idx": 0,
        "preset_key": next(iter(PRESETS)),
        "rotation_enabled": True,
        "plans": None,
        "click_targets": {},  # idx -> "anchor" | "scale"
        "last_click": {},  # idx -> last handled (x, y) to avoid reprocessing reruns
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def draw_overlay(image_rgb: np.ndarray, record: AlignmentRecord) -> Image.Image:
    img = Image.fromarray(image_rgb).convert("RGB")
    draw = ImageDraw.Draw(img)
    radius = max(4, img.width // 150)

    def dot(point: tuple[float, float], color: str) -> None:
        x, y = point
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline="white")

    ax, ay = record.anchor
    rx, ry = record.reference_point
    draw.line((ax, ay, rx, ry), fill="yellow", width=max(2, radius // 2))
    dot(record.anchor, "lime")
    dot(record.reference_point, "deepskyblue")
    return img


def effective_record(idx: int) -> AlignmentRecord:
    if idx in st.session_state.manual_records:
        return st.session_state.manual_records[idx]
    return st.session_state.auto_records[idx][1]


def run_detection(detector: PoseDetector) -> None:
    progress = st.progress(0.0, text="Detecting poses...")
    total = len(st.session_state.frames)
    for idx, image in enumerate(st.session_state.frames):
        if idx not in st.session_state.auto_records:
            height, width = image.shape[:2]
            pose_result = detector.detect(image)
            if pose_result.detected:
                record, warnings = alignment_record_from_landmarks(pose_result.landmarks, width, height)
            else:
                record, warnings = alignment_record_from_landmarks({}, width, height)
            st.session_state.auto_records[idx] = (pose_result, record, warnings)
        progress.progress((idx + 1) / total, text=f"Detecting poses... ({idx + 1}/{total})")
    progress.empty()


def build_plans():
    preset = PRESETS[st.session_state.preset_key]
    overrides = {idx: effective_record(idx) for idx in range(len(st.session_state.frames))}
    detector = get_detector(str(MODEL_PATH))
    return pipeline.build_frame_plans(
        st.session_state.frames,
        preset,
        st.session_state.reference_frame_idx,
        detector,
        manual_overrides=overrides,
        rotation_enabled=st.session_state.rotation_enabled,
    )


def main() -> None:
    st.title("Sports Flipbook Aligner")
    st.caption(
        "Upload a swing sequence, pick where it's going, and export frames "
        "cropped/aligned to a consistent position so the carousel reads as smooth motion."
    )
    init_state()

    if not MODEL_PATH.exists():
        st.error(
            f"Pose model not found at `{MODEL_PATH}`. Run "
            "`python scripts/download_models.py` first, then reload this page."
        )
        return

    uploaded_files = st.file_uploader(
        "Photo series (upload in sequence order)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        signature = file_signature(uploaded_files)
        if signature != st.session_state.upload_signature:
            st.session_state.upload_signature = signature
            st.session_state.frames = [load_frame(f) for f in uploaded_files]
            st.session_state.frame_names = [f.name for f in uploaded_files]
            st.session_state.auto_records = {}
            st.session_state.manual_records = {}
            st.session_state.reference_frame_idx = 0
            st.session_state.plans = None

    if not st.session_state.frames:
        st.info("Upload at least one photo to get started.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        preset_key = st.radio(
            "Platform / format",
            options=list(PRESETS.keys()),
            format_func=lambda k: PRESETS[k].label,
            index=list(PRESETS.keys()).index(st.session_state.preset_key),
        )
        st.session_state.preset_key = preset_key
    with col2:
        ref_idx = st.selectbox(
            "Reference frame (address / neutral pose)",
            options=list(range(len(st.session_state.frames))),
            format_func=lambda i: st.session_state.frame_names[i],
            index=st.session_state.reference_frame_idx,
        )
        st.session_state.reference_frame_idx = ref_idx
    with col3:
        st.session_state.rotation_enabled = st.checkbox(
            "Correct camera tilt (rotation)", value=st.session_state.rotation_enabled
        )

    detector = get_detector(str(MODEL_PATH))

    if st.button("Run pose detection", type="primary"):
        run_detection(detector)

    missing = [i for i in range(len(st.session_state.frames)) if i not in st.session_state.auto_records]
    if missing:
        st.warning("Run pose detection before reviewing/exporting.")
        return

    st.session_state.plans = build_plans()

    st.subheader("Review & correct")
    st.caption(
        "Click a thumbnail to set the point for the selected click target. "
        "Green = anchor (hip). Blue = scale/rotation reference (shoulder)."
    )

    cols = st.columns(3)
    for idx, image in enumerate(st.session_state.frames):
        plan = st.session_state.plans[idx]
        col = cols[idx % 3]
        with col:
            st.markdown(f"**{st.session_state.frame_names[idx]}**")
            record = effective_record(idx)
            overlay = draw_overlay(image, record)
            scale = THUMBNAIL_WIDTH / overlay.width
            display_img = overlay.resize((THUMBNAIL_WIDTH, int(overlay.height * scale)))

            click_target = st.radio(
                "Click sets",
                options=["anchor", "scale"],
                key=f"click_target_{idx}",
                horizontal=True,
                label_visibility="collapsed",
            )

            click = streamlit_image_coordinates(display_img, key=f"click_{idx}")
            if click is not None:
                coords = (click["x"], click["y"])
                if st.session_state.last_click.get(idx) != coords:
                    st.session_state.last_click[idx] = coords
                    orig_x, orig_y = coords[0] / scale, coords[1] / scale
                    current = effective_record(idx)
                    if click_target == "anchor":
                        new_record = AlignmentRecord(
                            anchor=(orig_x, orig_y),
                            reference_point=current.reference_point,
                            source="manual",
                            confidence=1.0,
                            needs_manual=False,
                        )
                    else:
                        new_record = AlignmentRecord(
                            anchor=current.anchor,
                            reference_point=(orig_x, orig_y),
                            source="manual",
                            confidence=1.0,
                            needs_manual=False,
                        )
                    st.session_state.manual_records[idx] = new_record
                    st.rerun()

            if idx in st.session_state.manual_records:
                if st.button("Reset to auto-detected", key=f"reset_{idx}"):
                    del st.session_state.manual_records[idx]
                    st.rerun()

            for warning in plan.warnings:
                st.caption(f":warning: {warning}")

            if plan.output_image is not None:
                preview = Image.fromarray(plan.output_image)
                preview_scale = PREVIEW_WIDTH / preview.width
                preview = preview.resize((PREVIEW_WIDTH, int(preview.height * preview_scale)))
                st.image(preview, caption="Aligned output preview")

    st.subheader("Export")
    exportable = pipeline.exportable_plans(st.session_state.plans)
    blocked = len(st.session_state.plans) - len(exportable)
    if blocked:
        st.warning(f"{blocked} frame(s) still need manual correction and will be excluded from export.")

    if exportable:
        preset = PRESETS[st.session_state.preset_key]
        zip_bytes = export.build_zip(exportable, preset, len(st.session_state.frames))
        st.download_button(
            "Download aligned carousel (.zip)",
            data=zip_bytes,
            file_name=f"flipbook_{preset.key}.zip",
            mime="application/zip",
        )
    else:
        st.info("No frames are ready to export yet.")


if __name__ == "__main__":
    main()
