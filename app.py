"""Streamlit entrypoint: upload a photo series, pick a platform/format, review
and correct the alignment, then export cropped/aligned stills.

Only the designated reference frame is pose-detected (to establish the
crop's framing); every other frame is aligned by matching its static
background to that reference frame rather than tracking the moving subject,
so the crop only has to cancel out camera shake/pan between shots.
"""

from __future__ import annotations

import hashlib

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageOps
from streamlit_image_coordinates import streamlit_image_coordinates

from flipbook import export, features, pipeline
from flipbook.detection import PoseDetector, detect_frame
from flipbook.model_assets import ensure_model
from flipbook.models import AlignmentRecord
from flipbook.presets import PRESETS

THUMBNAIL_WIDTH = 420
PREVIEW_WIDTH = 320

# Long-lens sports cameras commonly shoot 20-45MP JPEGs, which decompress to
# 70-150MB+ raw arrays each -- holding a full swing sequence of those in
# memory at once can exceed hosting limits (e.g. Streamlit Community Cloud's
# free-tier 1GB RAM). Our largest output preset tops out at 1920px, so there's
# no quality reason to keep originals larger than this working resolution.
MAX_WORKING_DIMENSION = 2400

st.set_page_config(page_title="Flipbook Aligner", layout="wide")


@st.cache_resource(show_spinner="Preparing pose detection model (one-time, ~10MB download)...")
def get_detector() -> PoseDetector:
    model_path = ensure_model()
    return PoseDetector(model_path)


def load_frame(uploaded_file) -> np.ndarray:
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    longest_side = max(image.width, image.height)
    if longest_side > MAX_WORKING_DIMENSION:
        scale = MAX_WORKING_DIMENSION / longest_side
        new_size = (round(image.width * scale), round(image.height * scale))
        image = image.resize(new_size, Image.LANCZOS)

    return np.array(image)


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
        "preset_key": next(iter(PRESETS)),
        "reference_frame_idx": 0,
        "rotation_enabled": True,
        "reference_pose": None,  # (PoseResult, AlignmentRecord, warnings) for reference_frame_idx
        "reference_manual_record": None,
        "frame_manual_matrices": {},  # frame_idx -> np.ndarray (from manual point correspondences)
        "ref_click_points": [None, None],  # 2 points clicked on the reference's cropped preview
        "pending_frame_points": {},  # frame_idx -> [point_or_None, point_or_None]
        "last_click": {},  # dedup key -> last handled (x, y), avoids reprocessing stale reruns
        "_framing_signature": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_for_new_upload() -> None:
    st.session_state.reference_pose = None
    st.session_state.reference_manual_record = None
    st.session_state.frame_manual_matrices = {}
    st.session_state.ref_click_points = [None, None]
    st.session_state.pending_frame_points = {}


def reset_framing_dependent_state() -> None:
    """Anything keyed to output-canvas pixel coordinates goes stale whenever
    the preset, reference frame, or rotation toggle changes (they all change
    the reference frame's crop transform, and thus the max-area scale)."""
    st.session_state.ref_click_points = [None, None]
    st.session_state.pending_frame_points = {}
    st.session_state.frame_manual_matrices = {}


def draw_body_overlay(image_rgb: np.ndarray, record: AlignmentRecord) -> Image.Image:
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


def draw_point_overlay(image_rgb: np.ndarray, points: list) -> Image.Image:
    img = Image.fromarray(image_rgb).convert("RGB")
    draw = ImageDraw.Draw(img)
    radius = max(4, img.width // 150)
    for point, color in zip(points, ("lime", "deepskyblue")):
        if point is None:
            continue
        x, y = point
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline="white")
    return img


def ensure_reference_pose(detector: PoseDetector) -> None:
    if st.session_state.reference_pose is None:
        image = st.session_state.frames[st.session_state.reference_frame_idx]
        st.session_state.reference_pose = detect_frame(detector, image)


def get_reference_record() -> AlignmentRecord:
    if st.session_state.reference_manual_record is not None:
        return st.session_state.reference_manual_record
    return st.session_state.reference_pose[1]


def build_plans(detector: PoseDetector):
    preset = PRESETS[st.session_state.preset_key]
    return pipeline.build_frame_plans(
        st.session_state.frames,
        preset,
        st.session_state.reference_frame_idx,
        detector,
        reference_record=get_reference_record(),
        manual_frame_matrices=st.session_state.frame_manual_matrices,
        rotation_enabled=st.session_state.rotation_enabled,
    )


def main() -> None:
    st.title("Sports Flipbook Aligner")
    st.caption(
        "Upload a swing sequence, pick where it's going, and export frames "
        "cropped/aligned to a consistent position so the carousel reads as smooth motion. "
        "Frames are aligned to the reference photo's stationary background (trees, golf bag, "
        "ground), not the moving golfer, so the crop can stay loose and consistent."
    )
    init_state()

    try:
        detector = get_detector()
    except Exception as exc:  # noqa: BLE001 - surfaced to the user, not a bug to swallow
        st.error(f"Couldn't load the pose detection model: {exc}")
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
            st.session_state.reference_frame_idx = 0
            reset_for_new_upload()

    if not st.session_state.frames:
        st.info("Upload at least one photo to get started.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.session_state.preset_key = st.radio(
            "Platform / format",
            options=list(PRESETS.keys()),
            format_func=lambda k: PRESETS[k].label,
            index=list(PRESETS.keys()).index(st.session_state.preset_key),
        )
    with col2:
        new_ref_idx = st.selectbox(
            "Reference frame (address / neutral pose)",
            options=list(range(len(st.session_state.frames))),
            format_func=lambda i: st.session_state.frame_names[i],
            index=st.session_state.reference_frame_idx,
        )
        if new_ref_idx != st.session_state.reference_frame_idx:
            st.session_state.reference_frame_idx = new_ref_idx
            st.session_state.reference_pose = None
            st.session_state.reference_manual_record = None
    with col3:
        st.session_state.rotation_enabled = st.checkbox(
            "Correct camera tilt (rotation)", value=st.session_state.rotation_enabled
        )

    framing_signature = (
        st.session_state.preset_key,
        st.session_state.reference_frame_idx,
        st.session_state.rotation_enabled,
    )
    if st.session_state._framing_signature != framing_signature:
        st.session_state._framing_signature = framing_signature
        reset_framing_dependent_state()

    if st.button("Run alignment", type="primary"):
        ensure_reference_pose(detector)

    if st.session_state.reference_pose is None:
        st.warning("Click 'Run alignment' to detect the reference frame's pose and align the series.")
        return

    ref_record = get_reference_record()

    st.subheader("Reference frame")
    st.caption(
        "This frame's pose sets the crop framing for the whole series -- correct it here if the "
        "auto-detected points look wrong."
    )
    ref_idx = st.session_state.reference_frame_idx
    ref_image = st.session_state.frames[ref_idx]
    ref_col = st.columns(3)[0]
    with ref_col:
        overlay = draw_body_overlay(ref_image, ref_record)
        ref_scale = THUMBNAIL_WIDTH / overlay.width
        display_img = overlay.resize((THUMBNAIL_WIDTH, int(overlay.height * ref_scale)))

        click_target = st.radio(
            "Click sets",
            options=["anchor", "scale"],
            key="ref_click_target",
            horizontal=True,
            label_visibility="collapsed",
        )
        click = streamlit_image_coordinates(display_img, key="ref_click")
        if click is not None:
            coords = (click["x"], click["y"])
            if st.session_state.last_click.get("ref") != coords:
                st.session_state.last_click["ref"] = coords
                orig = (coords[0] / ref_scale, coords[1] / ref_scale)
                current = ref_record
                if click_target == "anchor":
                    new_record = AlignmentRecord(
                        anchor=orig, reference_point=current.reference_point,
                        source="manual", confidence=1.0, needs_manual=False,
                    )
                else:
                    new_record = AlignmentRecord(
                        anchor=current.anchor, reference_point=orig,
                        source="manual", confidence=1.0, needs_manual=False,
                    )
                st.session_state.reference_manual_record = new_record
                reset_framing_dependent_state()
                st.rerun()

        if st.session_state.reference_manual_record is not None:
            if st.button("Reset reference to auto-detected"):
                st.session_state.reference_manual_record = None
                reset_framing_dependent_state()
                st.rerun()

        for warning in st.session_state.reference_pose[2]:
            st.caption(f":warning: {warning}")

    ref_record = get_reference_record()
    if ref_record.needs_manual:
        st.warning(
            "Set the reference frame's anchor and scale points above before the rest of the "
            "series can be aligned."
        )
        return

    plans = build_plans(detector)

    blocked = [p for p in plans if p.blocked]
    if blocked:
        st.subheader("Fix unmatched frames")
        st.caption(
            "These frames' backgrounds couldn't be auto-matched to the reference frame. Click the "
            "same two distinct, stationary points (e.g. a corner of the golf bag, a tree trunk "
            "base) in the reference preview below, then click those same two points in each frame "
            "that needs fixing."
        )
        point_choice = st.radio(
            "Click sets", options=["Point 1", "Point 2"], key="point_slot_selector", horizontal=True
        )
        slot_idx = 0 if point_choice == "Point 1" else 1

        st.markdown("**Reference points** (click on the aligned reference preview)")
        ref_plan = next(p for p in plans if p.frame_index == ref_idx)
        ref_preview_overlay = draw_point_overlay(ref_plan.output_image, st.session_state.ref_click_points)
        rp_scale = PREVIEW_WIDTH / ref_preview_overlay.width
        ref_preview_display = ref_preview_overlay.resize(
            (PREVIEW_WIDTH, int(ref_preview_overlay.height * rp_scale))
        )
        ref_point_click = streamlit_image_coordinates(ref_preview_display, key="ref_point_click")
        if ref_point_click is not None:
            coords = (ref_point_click["x"], ref_point_click["y"])
            if st.session_state.last_click.get("ref_points") != coords:
                st.session_state.last_click["ref_points"] = coords
                orig = (coords[0] / rp_scale, coords[1] / rp_scale)
                st.session_state.ref_click_points[slot_idx] = orig
                st.rerun()
        if st.button("Clear reference points"):
            st.session_state.ref_click_points = [None, None]
            st.rerun()

        cols = st.columns(3)
        for i, plan in enumerate(blocked):
            idx = plan.frame_index
            col = cols[i % 3]
            with col:
                st.markdown(f"**{st.session_state.frame_names[idx]}**")
                for warning in plan.warnings:
                    st.caption(f":warning: {warning}")
                points = st.session_state.pending_frame_points.setdefault(idx, [None, None])
                overlay = draw_point_overlay(st.session_state.frames[idx], points)
                scale = THUMBNAIL_WIDTH / overlay.width
                display_img = overlay.resize((THUMBNAIL_WIDTH, int(overlay.height * scale)))
                click = streamlit_image_coordinates(display_img, key=f"frame_point_click_{idx}")
                if click is not None:
                    coords = (click["x"], click["y"])
                    dedup_key = f"frame_points_{idx}"
                    if st.session_state.last_click.get(dedup_key) != coords:
                        st.session_state.last_click[dedup_key] = coords
                        orig = (coords[0] / scale, coords[1] / scale)
                        points[slot_idx] = orig
                        st.session_state.pending_frame_points[idx] = points
                        if all(p is not None for p in points) and all(
                            p is not None for p in st.session_state.ref_click_points
                        ):
                            matrix = features.fit_transform_from_point_pairs(
                                points, st.session_state.ref_click_points
                            )
                            if matrix is not None:
                                st.session_state.frame_manual_matrices[idx] = matrix
                        st.rerun()

        plans = build_plans(detector)

    st.subheader("Aligned previews")
    cols = st.columns(3)
    for i, plan in enumerate(plans):
        if plan.blocked:
            continue
        col = cols[i % 3]
        with col:
            st.markdown(f"**{st.session_state.frame_names[plan.frame_index]}**")
            if plan.match_info:
                st.caption(plan.match_info)
            for warning in plan.warnings:
                st.caption(f":warning: {warning}")
            if plan.output_image is not None:
                preview = Image.fromarray(plan.output_image)
                preview_scale = PREVIEW_WIDTH / preview.width
                preview = preview.resize((PREVIEW_WIDTH, int(preview.height * preview_scale)))
                st.image(preview)

    st.subheader("Export")
    exportable = pipeline.exportable_plans(plans)
    still_blocked = len(plans) - len(exportable)
    if still_blocked:
        st.warning(f"{still_blocked} frame(s) still need manual correction and will be excluded from export.")

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
