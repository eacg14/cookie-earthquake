"""Streamlit entrypoint: upload a photo series, pick a platform/format, review
and correct the alignment, then export cropped/aligned stills.

The designated reference frame is used exactly as shot -- a plain center-crop
to the target aspect ratio, no repositioning, no rotation leveling. Every
other frame is aligned to it by matching its static background rather than
tracking or recentering the subject, so the crop only cancels out camera
shake/pan between shots and never overrides how the photographer framed the
shot.
"""

from __future__ import annotations

import hashlib

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageOps

from streamlit_image_coordinates import streamlit_image_coordinates

from flipbook import export, features, pipeline
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
    st.session_state.frame_manual_matrices = {}
    st.session_state.ref_click_points = [None, None]
    st.session_state.pending_frame_points = {}


def reset_framing_dependent_state() -> None:
    """Anything keyed to output-canvas pixel coordinates goes stale whenever
    the preset or reference frame changes (both change the reference frame's
    crop transform, and thus the max-area scale)."""
    st.session_state.ref_click_points = [None, None]
    st.session_state.pending_frame_points = {}
    st.session_state.frame_manual_matrices = {}


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


def build_plans():
    preset = PRESETS[st.session_state.preset_key]
    return pipeline.build_frame_plans(
        st.session_state.frames,
        preset,
        st.session_state.reference_frame_idx,
        manual_frame_matrices=st.session_state.frame_manual_matrices,
    )


def main() -> None:
    st.title("Sports Flipbook Aligner")
    st.caption(
        "Upload a swing sequence, pick where it's going, and export frames "
        "cropped/aligned to a consistent position so the carousel reads as smooth motion. "
        "The reference photo's framing is kept exactly as shot; every other frame is aligned "
        "to its stationary background (trees, golf bag, ground), not the moving golfer, so the "
        "crop stays consistent without overriding how you composed the shot."
    )
    init_state()

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

    col1, col2 = st.columns(2)
    with col1:
        st.session_state.preset_key = st.radio(
            "Platform / format",
            options=list(PRESETS.keys()),
            format_func=lambda k: PRESETS[k].label,
            index=list(PRESETS.keys()).index(st.session_state.preset_key),
        )
    with col2:
        st.session_state.reference_frame_idx = st.selectbox(
            "Reference frame (sets the framing for the whole series, used exactly as shot)",
            options=list(range(len(st.session_state.frames))),
            format_func=lambda i: st.session_state.frame_names[i],
            index=st.session_state.reference_frame_idx,
        )

    framing_signature = (st.session_state.preset_key, st.session_state.reference_frame_idx)
    if st.session_state._framing_signature != framing_signature:
        st.session_state._framing_signature = framing_signature
        reset_framing_dependent_state()

    plans = build_plans()
    ref_idx = st.session_state.reference_frame_idx

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

        plans = build_plans()

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
