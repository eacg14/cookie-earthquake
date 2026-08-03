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
3. Pick a "reference frame" (usually address) -- it anchors the camera-tilt
   correction applied uniformly across the sequence.
4. Run pose detection. Each frame gets an auto-detected anchor (hip midpoint)
   and scale/rotation reference (shoulder midpoint), with fallbacks for
   occluded/side-on poses.
5. Review the grid: click any thumbnail to correct a bad anchor or
   scale/rotation point by hand. Frames MediaPipe couldn't read at all are
   blocked from export until corrected.
6. Export a zip of the cropped/aligned stills, ready to upload as a carousel.

See `/root/.claude/plans/i-take-sports-photos-greedy-emerson.md`-style design
notes in the module docstrings (`flipbook/alignment.py` especially) for the
reasoning behind the alignment math.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_models.py   # fetches the MediaPipe pose model (~30MB), one-time
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

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
resolution, and target anchor/scale framing. Nothing else needs to change --
`alignment.py`, `cropping.py`, and `pipeline.py` all consume `PresetSpec`
generically.
