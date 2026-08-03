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
3. Pick a "reference frame" (usually address). It's used **exactly as shot**
   -- no repositioning, no rotation leveling -- and defines the series'
   canonical framing: whatever composition the photographer chose is kept.
4. Every other frame is aligned to it by matching its *static background*
   (trees, golf bag, ground texture) via ORB feature matching + RANSAC, not
   by tracking or recentering the moving golfer. This corrects handheld
   camera shake/pan between shots -- both translation and rotation -- without
   ever deciding where the subject "should" sit in frame.
5. The crop's size is chosen automatically: once every frame is registered to
   the reference frame's coordinate space, the app finds the largest crop
   that stays within *every* frame's real pixel bounds simultaneously -- the
   maximum area shared by the whole aligned series, with zero fabricated
   (border-replicated) pixels. There's nothing to tune by hand.
6. Review: frames whose background couldn't be reliably matched are flagged
   -- click 2 matching static points (e.g. a golf bag corner) in the
   reference preview and in the frame that needs fixing.
7. Export a zip of the cropped/aligned stills, ready to upload as a carousel.

## Setup (local)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

## Deploying to Streamlit Community Cloud

This app is **not** deployable on Vercel: Streamlit is a long-running server
that holds a WebSocket connection open per user for its reactive UI (session
state, the click-to-correct grid, live previews), which doesn't fit Vercel's
serverless-function model at all. [Streamlit Community
Cloud](https://streamlit.io/cloud) is built for exactly this repo shape and
is free:

1. Push this repo to GitHub (already done if you're reading this from here).
2. On share.streamlit.io, create a new app pointing at this repo/branch and `app.py`.
3. Deploy. `requirements.txt` is picked up automatically -- no system
   (`packages.txt`) dependencies are needed; `opencv-python-headless` is a
   genuinely GUI-free build.

## Development

```bash
pip install -r requirements-dev.txt
pytest
```

`flipbook/alignment.py`, `flipbook/presets.py`, `flipbook/cropping.py`, and
`flipbook/features.py` are pure-math modules with no Streamlit dependency,
and are where the unit tests live (including a synthetic camera-shake
recovery test for background matching). The Streamlit UI itself is verified
by hand.

## Adding a new platform/format

Add a row to `PRESETS` in `flipbook/presets.py` with the aspect ratio and
output resolution. Nothing else needs to change -- `alignment.py`,
`cropping.py`, and `pipeline.py` all consume `PresetSpec` generically, and
crop position/size are always derived from the photos themselves (the
reference frame's own framing, plus the max-area search), never a preset
constant.
