# Image Detection Bypass Utility

Circumvention of AI Detection — all wrapped in a clean, user-friendly interface.

![MIT License](https://img.shields.io/badge/License-MIT-green.svg)
![Python 3.13](https://img.shields.io/badge/Python-3.13-yellow)
![PyQt5](https://img.shields.io/badge/PyQt5-latest-blue)
![Pillow](https://img.shields.io/badge/Pillow-latest-lightgrey)
![NumPy](https://img.shields.io/badge/NumPy-latest-lightgrey)
![Matplotlib](https://img.shields.io/badge/Matplotlib-latest-lightgrey)
![piexif](https://img.shields.io/badge/piexif-latest-lightgrey)
![SciPy](https://img.shields.io/badge/SciPy-latest-lightgrey)
![OpenCV](https://img.shields.io/badge/OpenCV-latest-darkgreen)
![Torch 2.8.0+cu126](https://img.shields.io/badge/Torch-2.8.0%2Bcu126-red)
![TorchVision 0.23.0+cu126](https://img.shields.io/badge/TorchVision-0.23.0%2Bcu126-orange)


---

## Screenshot

![Screenshot](https://i.imgur.com/y0tuDcK.png)

## Notice
Due to the nature of this project, future updates will be under AGPL V3 license to ensure this project and its derivatives remains Open Source.

## Features

* Select input, optional auto white-balance reference, optional FFT reference, and output paths with live previews.
* **Auto Mode**: one slider to control an expressive preset of postprocess parameters.
* **Manual Mode**: full access to noise, CLAHE, FFT, phase perturbation, pixel perturbation, etc.
* Camera pipeline simulator: Bayer/demosaic, JPEG cycles/quality, vignette, chromatic aberration, motion blur, hot pixels, read-noise, banding.
* Input / output analysis panels (via `AnalysisPanel`) to inspect images before/after processing.
* Background worker thread with progress reporting and rich error dialog (traceback viewer).

---

## ComfyUI Integration

![Screenshot](https://i.imgur.com/KzjEfxf.png)

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
pip install pyqt5 pillow numpy matplotlib piexif lpips
# optional but recommended for extra functionality:
pip install opencv-python
# optional but needed for AI Normalizer (Install CPU OR Cuda)
#Torch Cuda 12.6
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
#Torch CPU
pip install torch torchvision

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

# Alternatively, if you're having issues, a "run.sh" script has been created that will also install dependencies 
# properly  and run the GUI
./run.sh # also installs dependencies before running `python run.py`
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

## AI Normalizer

When enabled, the AI Normalizer applies a non-semantic attack using PyTorch and LPIPS to subtly modify the image without introducing perceptible artifacts. The following parameters control its behavior:

* **Iterations** — Number of optimization steps to perform.
* **Learning Rate** — Step size for the optimizer.
* **T LPIPS** — Threshold for the LPIPS perceptual loss. If the LPIPS loss exceeds this threshold, it is penalized.
* **T L2** — Threshold for the L2 loss on the perturbation.
* **C LPIPS** — Weighting factor for the LPIPS loss penalty.
* **C L2** — Weighting factor for the L2 loss penalty.
* **Gradient Clip** — Maximum allowed gradient value during optimization to prevent exploding gradients.

---

## Contributing

* PRs welcome. If you modify UI layout or parameter names, keep the `args` mapping consistent or update `README` and `worker.py` accordingly.
* Add unit tests for `worker.py` and the parameter serialization if you intend to refactor.

---

## Paper Used

This project credits and draws inspiration from:

**UnMarker: A Universal Attack on Defensive Image Watermarking**  
Andre Kassis, Urs Hengartner

## License

MIT — free to use and adapt. Please include attribution if you fork or republish.

---
