# Sports Flipbook Aligner

A local web app for sports photographers (built with golf swings in mind) that
takes a series of hand-held, long-lens photos and outputs them cropped and
aligned to a consistent position/size/orientation -- so a swing sequence reads
as smooth motion when posted as an Instagram carousel, instead of a jittery,
zoom-drifting mess.

## How it works

1. Upload a photo series in sequence order (e.g. address, backswing, top,
   downswing, impact, follow-through).
2. Pick the target format: Instagram Feed Square (1:1), Feed Portrait (4:5),
   or Stories/Reels (9:16).
3. Pick a "reference frame" (usually address) and run alignment. Only this
   frame gets pose-detected (hip anchor + shoulder/rotation reference, with
   fallbacks for occluded/side-on poses) -- it establishes the crop's
   position and camera-tilt correction for the whole series.
4. Every other frame is aligned by matching its *static background* (trees,
   golf bag, ground texture) to the reference frame via ORB feature matching
   + RANSAC, not by tracking the moving golfer. The crop only has to cancel
   out handheld camera shake between shots, so the swing moves naturally
   within a steady frame instead of the frame chasing the subject.
5. The crop's size is chosen automatically: once every frame is registered
   to the reference frame's coordinate space, the app finds the largest crop
   that stays within *every* frame's real pixel bounds simultaneously -- the
   maximum area shared by the whole aligned series, with zero fabricated
   (border-replicated) pixels. There's no manual zoom to tune.
6. Review: frames whose background couldn't be reliably matched are flagged
   -- click 2 matching static points (e.g. a golf bag corner) in the
   reference preview and in the frame that needs fixing. The reference
   frame's own anchor/scale points are also correctable by hand if pose
   detection got them wrong.
7. Export a zip of the cropped/aligned stills, ready to upload as a carousel.

See `/root/.claude/plans/i-take-sports-photos-greedy-emerson.md`-style design
notes in the module docstrings (`flipbook/alignment.py` especially) for the
reasoning behind the alignment math.

## Setup (local)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_models.py   # fetches the MediaPipe pose model (~10MB), one-time
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

The model download step is optional -- `app.py` fetches it automatically on
first use if it's missing -- but running it ahead of time avoids a delay on
your first page load.

## Deploying to Streamlit Community Cloud

This app is **not** deployable on Vercel: Streamlit is a long-running server
that holds a WebSocket connection open per user for its reactive UI (session
state, the click-to-correct grid, live previews), which doesn't fit Vercel's
serverless-function model at all. [Streamlit Community
Cloud](https://streamlit.io/cloud) is built for exactly this repo shape and
is free:

1. Push this repo to GitHub (already done if you're reading this from there).
2. On share.streamlit.io, create a new app pointing at this repo/branch and `app.py`.
3. In "Advanced settings", pick Python 3.11 (what this app is tested against).
4. Deploy. `requirements.txt` (pip deps) and `packages.txt` (apt deps --
   `libgl1`/`libglib2.0-0t64`, needed by OpenCV/MediaPipe) are picked up
   automatically. The pose model downloads itself on first use.

## Development

```bash
pip install -r requirements-dev.txt
pytest
```

`flipbook/alignment.py`, `flipbook/presets.py`, and `flipbook/cropping.py` are
pure-math/pure-data modules with no MediaPipe or Streamlit dependency, and are
where the unit tests live. Pose-detection accuracy on real photos and the
Streamlit UI itself are verified by hand -- see the plan file's verification
checklist.

## Adding a new platform/format

Add a row to `PRESETS` in `flipbook/presets.py` with the aspect ratio, output
resolution, and target anchor position. Nothing else needs to change --
`alignment.py`, `cropping.py`, and `pipeline.py` all consume `PresetSpec`
generically, and crop size is always computed automatically per-series.
