# Image Postprocessing Pipeline

This document covers the standalone image processing pipeline implemented in `image_postprocess/processor.py`.
It intentionally ignores the GUI/frontend and ComfyUI/node integration layers.

## Entry Point

Run the pipeline from the repository root as a module:

```bash
python -m image_postprocess.processor input.jpg output.jpg [options]
```

The module expects two positional arguments:

* `input` - source image path
* `output` - destination image path

The pipeline always loads the input as RGB, processes it in memory, adds fake EXIF metadata, and saves the result.

## Processing Order

When multiple effects are enabled, the pipeline applies them in this order:

1. Load the input image as RGB
2. Load FFT reference image if `--fft-ref` is provided
3. Blend colors with k-means-based region reduction if `--blend` is enabled
4. Apply non-semantic attack if `--non-semantic` is enabled
5. Apply FFT spectral matching if `--fft` is enabled
6. Apply GLCM normalization if `--glcm` is enabled
7. Add Gaussian noise if `--noise` is enabled
8. Apply randomized perturbation if `--perturb` is enabled
9. Run the camera simulator if `--sim-camera` is enabled
10. Apply auto white balance if `--awb` is enabled
11. Apply LUT mapping if `--lut` is provided
12. Save the output with generated fake EXIF data

## Basic Usage

Minimal conversion:

```bash
python -m image_postprocess.processor input.png output.png
```

Example with several effects enabled:

```bash
python -m image_postprocess.processor input.jpg output.jpg \
  --noise --fft --fft-ref reference.jpg \
  --sim-camera --awb --lut look.cube
```

## Supported Features

### Auto White Balance

Enable with `--awb`.

* If `--ref` is provided, the pipeline matches the image mean color to the reference image mean color.
* If `--ref` is omitted, it falls back to a gray-world assumption.

Options:

* `--ref PATH` - optional AWB reference image

### FFT Spectral Matching

Enable with `--fft`.

FFT can work in reference mode or model mode:

* `--fft-ref PATH` - optional reference image used for FFT matching
* `--fft-mode {auto,ref,model}` - `auto` uses the reference image if present, otherwise model mode
* `--fft-alpha FLOAT` - slope parameter for the model spectrum, default `1.0`
* `--cutoff FLOAT` - low-frequency cutoff, default `0.25`
* `--fstrength FLOAT` - blend strength, default `0.9`
* `--randomness FLOAT` - multiplicative spectral randomness, default `0.05`
* `--phase-perturb FLOAT` - phase noise strength in radians, default `0.08`
* `--radial-smooth INT` - radial smoothing bins, default `5`

`--fft-ref` is also reused by the GLCM stage.

### GLCM Normalization

Enable with `--glcm`.

This stage matches second-order texture statistics on the luminance channel while preserving chroma.

Options:

* `--glcm-distances INT [INT ...]` - pixel offsets for co-occurrence matrices, default `1`
* `--glcm-angles FLOAT [FLOAT ...]` - angles in radians, default `0`, `pi/4`, `pi/2`, `3*pi/4`
* `--glcm-levels INT` - gray levels, default `256`
* `--glcm-strength FLOAT` - blend strength, default `0.9`

### Gaussian Noise

Enable with `--noise`.

* `--noise-std FLOAT` - Gaussian noise standard deviation as a fraction of 255, default `0.02`

### Randomized Perturbation

Enable with `--perturb`.

* `--perturb-magnitude FLOAT` - per-pixel perturbation magnitude fraction, default `0.008`

### Non-Semantic Attack

Enable with `--non-semantic`.

This uses the PyTorch/LPIPS-based attack path when the optional dependencies are available.

Controls:

* `--ns-iterations INT` - optimization steps, default `500`
* `--ns-learning-rate FLOAT` - optimizer learning rate, default `3e-4`
* `--ns-t-lpips FLOAT` - LPIPS threshold, default `4e-2`
* `--ns-t-l2 FLOAT` - L2 threshold, default `3e-5`
* `--ns-c-lpips FLOAT` - LPIPS penalty constant, default `1e-2`
* `--ns-c-l2 FLOAT` - L2 penalty constant, default `0.6`
* `--ns-grad-clip FLOAT` - gradient clip value, default `0.05`

### Camera Simulation

Enable with `--sim-camera`.

This adds realistic capture artifacts such as Bayer/demosaic behavior, chromatic aberration, vignette, sensor noise, hot pixels, banding, motion blur, and JPEG recompression.

Controls:

* `--no-no-bayer` - disable Bayer/demosaic simulation
* `--jpeg-cycles INT` - number of JPEG recompression passes, default `1`
* `--jpeg-qmin INT` - minimum JPEG quality, default `88`
* `--jpeg-qmax INT` - maximum JPEG quality, default `96`
* `--vignette-strength FLOAT` - vignette amount, default `0.35`
* `--chroma-strength FLOAT` - chromatic aberration strength in pixels, default `1.2`
* `--iso-scale FLOAT` - ISO/exposure scale for Poisson noise, default `1.0`
* `--read-noise FLOAT` - read-noise sigma, default `2.0`
* `--hot-pixel-prob FLOAT` - per-pixel hot-pixel probability, default `1e-6`
* `--banding-strength FLOAT` - horizontal banding amplitude, default `0.0`
* `--motion-blur-kernel INT` - motion blur kernel size, default `1`

### LUT Application

Provide `--lut PATH` to apply a color lookup table after all other image transforms and after auto white balance.

Supported LUT formats:

* `.npy` - NumPy array saved with `numpy.save`
* `.cube` - 3D LUT file
* Image-based 1D LUTs - typically `256x1` or `1x256` PNG/JPG strips, or other flattened 1D LUT image layouts

Blend control:

* `--lut-strength FLOAT` - blend amount, default `0.1`

## Blending

Enable with `--blend`.

* `--blend-tolerance FLOAT` - color distance threshold, default `10.0`
* `--blend-min-region INT` - minimum retained region size, default `50`
* `--blend-max-samples INT` - maximum sampled pixels for k-means, default `100000`
* `--blend-n-jobs INT` - worker thread count, default `None`

## Randomness and Reproducibility

* `--seed INT` - shared seed for deterministic runs when supported by the selected stages

If `--seed` is omitted, each run may produce different results.

## Notes

* The output image is always saved with generated fake EXIF metadata.
* The pipeline assumes RGB input and internally converts images to NumPy arrays for processing.
* Some stages are optional and depend on extra packages such as OpenCV, SciPy, scikit-image, PyTorch, and LPIPS.
* If a stage fails at runtime, the processor prints a warning and skips that stage rather than aborting the full pipeline.

## Python API

The pipeline is also exposed as `image_postprocess.process_image(input_path, output_path, args)`.

The `args` object should provide the same attributes as the CLI parser, so the easiest way to call it from Python is to reuse `build_argparser()` and parse a synthetic argument list.

Example:

```python
from image_postprocess.processor import build_argparser, process_image

args = build_argparser().parse_args([
    "input.jpg",
    "output.jpg",
    "--fft",
    "--sim-camera",
])

process_image(args.input, args.output, args)
```
