# Image Detection Bypass Utility

A polished PyQt5 GUI for the `image_postprocess` pipeline that adds live previews, an input/output analysis panel, an optional camera simulator, and easy parameter control — all wrapped in a clean, user-friendly interface.

---

## Features
- Select input, optional reference, and output paths with live previews.  
- **Auto Mode**: one slider to control an expressive preset of postprocess parameters.  
- **Manual Mode**: full access to noise, CLAHE, FFT, phase perturbation, pixel perturbation, etc.  
- Camera pipeline simulator: Bayer/demosaic, JPEG cycles/quality, vignette, chromatic aberration, motion blur, hot pixels, read-noise, banding.  
- Input / output analysis panels (via `AnalysisPanel`) to inspect images before/after processing.  
- Background worker thread with progress reporting and rich error dialog (traceback viewer).  
- Graceful handling of `image_postprocess` import errors (shows a critical dialog with the import error).

---

## Quick start

### Requirements
- Python 3.8+ recommended  
- PyPI packages:
```bash
pip install pyqt5 pillow numpy matplotlib piexif
# optional but recommended for color matching / some functionality:
pip install opencv-python

OR

pip install -r requirements.txt
```

### Files expected in the same folder
- `image_postprocess.py` — your processing logic (export `process_image(...)` or compatible API).  
- `worker.py` — Worker thread wrapper used to run the pipeline in background.  
- `analysis_panel.py` — UI widget used for input/output analysis.  
- `utils.py` — must provide `qpixmap_from_path(path, max_size=(w,h))`.

### Run
Save the GUI script (for example) as `image_postprocess_gui.py` (or use the existing name `image_postprocess_gui_with_analysis_updated.py`) and run:

```bash
python3 image_postprocess_gui.py
```

If `image_postprocess` cannot be imported, the GUI will show an error explaining the import failure (see **Troubleshooting** below).

---

## Using the GUI (at-a-glance)
1. **Choose Input** — opens file dialog; sets suggested output path automatically.  
2. *(optional)* **Choose Reference** — used for FFT/color reference (OpenCV-based color match supported).  
3. **Choose Output** — where processed image will be written.  
4. **Auto Mode** — enable for a single slider to control a bundled preset.  
5. **Manual Mode** — tune individual parameters in the Parameters group.  
6. **Camera Simulator** — enable to reveal camera-specific controls (Bayer, JPEG cycles, vignette, chroma, etc.).  
7. Click **Run — Process Image** to start. The GUI disables controls while running and shows progress.  
8. When finished, the output preview and Output analysis panel update automatically.

---

## Parameters / Controls → `args` mapping

When you click **Run**, the GUI builds a lightweight argument namespace (similar to a `SimpleNamespace`) and passes it to the worker. Below are the important mappings used by the GUI (so you know what your `process_image` should expect):

- `args.noise_std` — Gaussian noise STD (fraction of 255)  
- `args.clahe_clip` — CLAHE clip limit  
- `args.tile` — CLAHE tile size  
- `args.cutoff` — Fourier cutoff (0.01–1.0)  
- `args.fstrength` — Fourier strength (0–1)  
- `args.phase_perturb` — phase perturbation STD (radians)  
- `args.randomness` — Fourier randomness factor  
- `args.perturb` — small pixel perturbations  
- `args.fft_mode` — one of `auto`, `ref`, `model`  
- `args.fft_alpha` — alpha exponent for 1/f model (used when `fft_mode=='model'`)  
- `args.radial_smooth` — radial smoothing bins for spectrum matching  
- `args.jpeg_cycles` — number of lossy JPEG encode/decode cycles (camera sim)  
- `args.jpeg_qmin`, `args.jpeg_qmax` — JPEG quality range used by camera sim  
- `args.vignette_strength` — vignette intensity (0–1)  
- `args.chroma_strength` — chromatic aberration strength (pixels)  
- `args.iso_scale` — exposure multiplier (camera sim)  
- `args.read_noise` — read noise in DN (camera sim)  
- `args.hot_pixel_prob` — probability of hot pixels (camera sim)  
- `args.banding_strength` — banding strength  
- `args.motion_blur_kernel` — motion blur kernel size  
- `args.seed` — integer seed or `None` when seed==0 in UI  
- `args.sim_camera` — bool: run camera simulation path  
- `args.no_no_bayer` — toggles Bayer/demosaic (True = enable RGGB demosaic)  
- `args.fft_ref` — path to reference image (string) or `None`

> **Tip:** Your `process_image(inpath, outpath, args)` should be tolerant of missing keys (use `getattr(args, 'name', default)`), or accept the same `SimpleNamespace` object the GUI builds.

---

## Error handling / UI behavior
- The GUI uses a `Worker` thread to avoid blocking the UI. Worker emits signals: `started`, `finished(outpath)`, `error(msg, traceback_text)`.  
- On error, a dialog displays the error message and a full traceback for debugging.  
- If `image_postprocess` fails to import at startup, a critical dialog shows the exception details; fix the module or dependencies and restart.

---

## Development notes
- **Integrating your pipeline:** make sure `image_postprocess.py` exports a `process_image(inpath, outpath, args)` function (or adapt `worker.py` to match your pipeline signature).  
- **Analysis panels:** `AnalysisPanel` should provide `update_from_path(path)`; used for both input and output.  
- **Preview helper:** `utils.qpixmap_from_path` is used to load scaled QPixmap for previews — useful for keeping UI responsive.
- **Packaging:** If you want a single executable, consider `PyInstaller` (note: include `worker.py`, `analysis_panel.py`, `utils.py`, and the pipeline module).

---

## Troubleshooting
- **ImportError for `image_postprocess`** — ensure `image_postprocess.py` is in the same directory or on `PYTHONPATH`. Also confirm required packages (numpy, Pillow, opencv) are installed. The GUI shows the import error text at startup.  
- **Previews blank/no image** — check that `qpixmap_from_path` returns a valid QPixmap. The preview widget falls back to `No image` if file missing.  
- **Processing hangs** — confirm Worker is implemented to emit `finished` or `error`. If your `process_image` blocks indefinitely, the GUI will appear unresponsive (worker runs in background thread but won't return).  
- **Color matching unavailable** — color matching uses OpenCV; if you did not install `opencv-python`, the GUI will still run, but reference-based color matching will be disabled.

---

## Example: minimal `process_image` signature
```python
# image_postprocess.py (sketch)
def process_image(inpath: str, outpath: str, args):
    # args is a SimpleNamespace with attributes described above
    # load image (PIL / numpy), run your pipeline, save output
    pass
```

---

## Contributing
- PRs welcome. If you modify UI layout or parameter names, keep the `args` mapping consistent or update `README` and `worker.py` accordingly.  
- Add unit tests for `worker.py` and the parameter serialization if you intend to refactor.

---

## License
MIT — free to use and adapt. Please include attribution if you fork or republish.

---
