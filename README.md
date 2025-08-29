# Image Detection Bypass Utility

Circumvention of AI Detection — all wrapped in a clean, user-friendly interface.

---

## Screenshot

![Screenshot](https://i.imgur.com/Jp9U8Rm.png)

## Features

* Select input, optional auto white-balance reference, optional FFT reference, and output paths with live previews.
* **Auto Mode**: one slider to control an expressive preset of postprocess parameters.
* **Manual Mode**: full access to noise, CLAHE, FFT, phase perturbation, pixel perturbation, etc.
* Camera pipeline simulator: Bayer/demosaic, JPEG cycles/quality, vignette, chromatic aberration, motion blur, hot pixels, read-noise, banding.
* Input / output analysis panels (via `AnalysisPanel`) to inspect images before/after processing.
* Background worker thread with progress reporting and rich error dialog (traceback viewer).

---

## Parameter Explanation

This section documents every manual parameter exposed by the GUI and gives guidance for usage.

---

## Manual Parameters

When **Auto Mode** is disabled, you can fine-tune the image post-processing pipeline manually using the following parameters:

### Noise & Contrast

* **Noise std (0–0.1)**
  Standard deviation of Gaussian noise applied to the image. Higher values introduce more noise, useful for simulating sensor artifacts.

* **CLAHE clip**
  Clip limit for Contrast Limited Adaptive Histogram Equalization (CLAHE). Controls the amount of contrast enhancement.

* **CLAHE tile**
  Number of tiles used in CLAHE grid. Larger values give finer local contrast adjustments.

---

### Fourier Domain Controls

* **Fourier cutoff (0–1)**
  Frequency cutoff threshold. Lower values preserve only low frequencies (smoothing), higher values retain more high-frequency detail.

* **Fourier strength (0–1)**
  Blending ratio for Fourier-domain filtering. At 1.0, full effect is applied; at 0.0, no effect.

* **Fourier randomness**
  Amount of stochastic variation introduced in the Fourier transform domain to simulate non-uniform distortions.

* **Phase perturb (rad)**
  Random perturbation of phase in the Fourier spectrum, measured in radians. Adds controlled irregularity to frequency response.

* **Radial smooth (bins)**
  Number of bins used for radial frequency smoothing. Higher values smooth the frequency response more aggressively.

* **FFT mode**
  Mode selection for FFT-based processing (e.g., `auto`, `ref`, `model`).
  `auto` will choose the most appropriate mode automatically.
  `ref` uses your FFT reference image as a reference.
  `model` uses a preset mathematical formula to find a natural FFT spectrum.

* **FFT alpha (model)**
  Scaling factor for FFT filtering. Controls how strongly frequency components are weighted. Only affects model mode.

---

### Pixel-Level Perturbations

* **Pixel perturb**
  Standard deviation of per-pixel perturbations applied in the spatial domain. Adds small jitter to pixel intensities.

---

### Texture-based Normalization (GLCM)

* **What it is** — Gray-Level Co-occurrence Matrix (GLCM) features capture second-order texture statistics (how often pairs of gray levels occur at specified distances and angles). GLCM normalization aims to match these texture statistics between the input and an FFT reference image, producing more realistic-looking sensor/textural artifacts.

* **When to use** — Use GLCM when the goal is to emulate or match textural properties (fine-grain patterns, structural noise) of a reference image. It is complementary to Fourier-domain matching: FFT shapes the spectral envelope while GLCM shapes spatial texture statistics.

* **Controls**

  * **Distances** — distance(s) in pixels used when building GLCM (default `[1]`). Use larger distances to capture coarser texture.
  * **Angles** — angles (radians) for directional co-occurrence (default `[0, π/4, π/2, 3π/4]`). More angles give rotation-robust matching.
  * **Levels** — number of quantized gray levels used for GLCM (default `256`). Lower values are faster and smooth the texture statistics.
  * **Strength** — blending factor (0..1) controlling how strongly GLCM features from the reference influence the output (default `0.9`). Lower values apply subtler texture matching.

* **Practical tips**

  * Combine moderate `glcm_strength` (0.4–0.8) with FFT matching for subtle realism.
  * Use fewer `glcm_levels` (e.g., 64 or 32) for speed and to avoid overfitting to noisy reference images.

---

### Local Binary Patterns (LBP)

* **What it is** — Local Binary Patterns encode a small neighbourhood around each pixel as a binary pattern, creating histograms that are very effective at characterizing micro-texture and local structure.

* **When to use** — LBP histogram matching is useful when you want to replicate micro-textural characteristics like sensor grain, cloth weave, or repetitive fine structure from a reference image.

* **Controls**

  * **Radius** — radius of the circular neighbourhood (in pixels) used to compute LBP (default `3`). Larger radii capture coarser patterns.
  * **N points** — number of sampling points around the circle (default `24`). More points increase descriptor resolution.
  * **Method** — one of `default`, `ror` (rotation invariant), `uniform` (compact uniform patterns), or `var` (variance-based). Use `uniform` for compact, robust histograms by default.
  * **Strength** — blending factor (0..1) controlling how strongly the LBP histogram from the reference influences the output (default `0.9`).

* **Practical tips**

  * Use `lbp_method='uniform'` and `lbp_n_points` 8–24 for stable results across natural images.
  * Decrease `lbp_strength` for subtle grain matching; increase it if the output needs to closely follow the reference micro-texture.

---

### Randomization

* **Seed (0=none)**
  Random seed for reproducibility.

  * `0` → fully random each run
  * Any other integer → deterministic output for given settings

---

Use these parameters to experiment with different looks.

Generally:
For **Minimum destructiveness**, keep noise and perturb values low.
For **Increased Evation**, increase Fourier randomness, Fourier Strength, phase perturb, and pixel perturb.

---

## ComfyUI Integration

Use ComfyUI Manager and install via GitHub link.
Or manually clone to custom\_nodes folder.

```bash
git clone https://github.com/PurinNyova/Image-Detection-Bypass-Utility
```

then

```bash
cd Image-Detection-Bypass-Utility
pip install -r requirements.txt
```

Thanks to u/Race88 for the help on the ComfyUI code.

### Requirements

* Python 3.8+ recommended
* PyPI packages:

```bash
pip install pyqt5 pillow numpy matplotlib piexif
# optional but recommended for color matching / some functionality:
pip install opencv-python

```

OR

```bash
pip install -r requirements.txt
```

### Files expected in the same folder

* `image_postprocess` — your processing logic (export `process_image(...)` or compatible API).
* `worker.py` — Worker thread wrapper used to run the pipeline in background.
* `analysis_panel.py` — UI widget used for input/output analysis.
* `utils.py` — must provide `qpixmap_from_path(path, max_size=(w,h))`.

### Run

```bash
python run.py
```

---

## Using the GUI (at-a-glance)

1. **Choose Input** — opens file dialog; sets suggested output path automatically.
2. *(optional)* **Choose Reference** — used for FFT/color reference (OpenCV-based color match supported).
3. *(optional)* **Choose Auto White-Balance Reference** — used for auto white-balance correction (applied before CLAHE).
4. **Choose Output** — where processed image will be written.
5. **Auto Mode** — enable for a single slider to control a bundled preset.
6. **Manual Mode** — tune individual parameters in the Parameters group.
7. **Camera Simulator** — enable to reveal camera-specific controls (Bayer, JPEG cycles, vignette, chroma, etc.).
8. Click **Run — Process Image** to start. The GUI disables controls while running and shows progress.
9. When finished, the output preview and Output analysis panel update automatically.

---

## Parameters / Controls → `args` mapping

When you click **Run**, the GUI builds a lightweight argument namespace (similar to a `SimpleNamespace`) and passes it to the worker. Below are the important mappings used by the GUI (so you know what your `process_image` should expect):

* `args.input` — Input image path (string)
* `args.output` — Output image path (string)
* `args.awb` — Toggle for automatic white balancing (bool, enables grey-world if `--ref` is not provided)
* `args.ref` — Path to auto white-balance reference image (string) or `None`
* `args.noise_std` — Gaussian noise standard deviation (fraction of 255, 0–0.1)
* `args.clahe_clip` — CLAHE clip limit for contrast enhancement
* `args.tile` — CLAHE tile grid size for contrast enhancement
* `args.cutoff` — Fourier cutoff frequency (0.01–1.0) for spectral processing
* `args.fstrength` — Fourier blend strength (0–1) for spectral processing
* `args.randomness` — Randomness factor for Fourier mask modulation
* `args.perturb` — Randomized pixel perturbation magnitude (fraction, 0–0.05)
* `args.seed` — Integer seed for reproducibility or `None` when seed==0 in UI
* `args.fft_ref` — Path to reference image for FFT spectral matching (string) or `None`
* `args.fft_mode` — FFT spectral matching mode: one of `auto`, `ref`, `model`
* `args.fft_alpha` — Alpha exponent for 1/f model (spectrum slope, used when `fft_mode=='model'`)
* `args.phase_perturb` — Phase perturbation standard deviation (radians) for FFT processing
* `args.radial_smooth` — Radial smoothing bins for spectrum profiles in FFT matching
* `args.sim_camera` — Bool: enables camera simulation pipeline (Bayer, CA, vignette, JPEG cycles)
* `args.no_no_bayer` — Bool: toggles Bayer/demosaic step (True = enable RGGB demosaic, False = disable)
* `args.jpeg_cycles` — Number of lossy JPEG encode/decode cycles for camera simulation
* `args.jpeg_qmin` — Minimum JPEG quality for recompression cycles
* `args.jpeg_qmax` — Maximum JPEG quality for recompression cycles
* `args.vignette_strength` — Vignette intensity (0–1) for camera simulation
* `args.chroma_strength` — Chromatic aberration strength (pixels) for camera simulation
* `args.iso_scale` — Exposure multiplier for Poisson noise in camera simulation
* `args.read_noise` — Read noise sigma in digital numbers (DN) for camera simulation
* `args.hot_pixel_prob` — Per-pixel probability of hot pixels in camera simulation
* `args.banding_strength` — Horizontal banding amplitude (0–1) for camera simulation
* `args.motion_blur_kernel` — Motion blur kernel size (1 = none) for camera simulation
* `args.lut` — Path to a 1D PNG (256x1), .npy LUT, or .cube 3D LUT (string) or `None`
* `args.lut_strength` — Strength to blend LUT (0.0 = no effect, 1.0 = full LUT)
* `args.glcm` — Bool: enable GLCM-based normalization (uses `args.fft_ref` when present).
* `args.glcm_distances` — List\[int]: distances used to compute GLCM (e.g., `[1]` or `[1,2,3]`).
* `args.glcm_angles` — List\[float]: angles (radians) used to compute GLCM (e.g., `[0, np.pi/4]`).
* `args.glcm_levels` — Int: number of quantized gray levels used when computing GLCM.
* `args.glcm_strength` — Float: blending strength for GLCM feature matching (0..1).
* `args.lbp` — Bool: enable LBP-based normalization (uses `args.fft_ref` when present).
* `args.lbp_radius` — Int: radius of circular neighbourhood for LBP.
* `args.lbp_n_points` — Int: number of sample points around the circle for LBP descriptor.
* `args.lbp_method` — Str: LBP computation method: `default`, `ror`, `uniform`, `var`.
* `args.lbp_strength` — Float: blending strength for LBP histogram matching (0..1).

---

## Contributing

* PRs welcome. If you modify UI layout or parameter names, keep the `args` mapping consistent or update `README` and `worker.py` accordingly.
* Add unit tests for `worker.py` and the parameter serialization if you intend to refactor.

---

## License

MIT — free to use and adapt. Please include attribution if you fork or republish.

---
