# FastAPI Backend Contract

## Goals

- Expose every master processing feature as its own HTTP endpoint.
- Expose one pipeline endpoint that can run multiple stages in one request.
- Preserve the existing default stage order when execution-order compatibility is disabled.
- Support execution-order compatibility by honoring the client-provided master-stage order for the multi-stage endpoint.
- Reuse the same processing core for CLI and API so behavior stays aligned.

## Master Stages

The backend recognizes the following stage identifiers:

- `blend`
- `non_semantic`
- `clahe`
- `fft`
- `glcm`
- `lbp`
- `noise`
- `perturb`
- `sim_camera`
- `awb`
- `lut`

Default stage order:

1. `blend`
2. `non_semantic`
3. `clahe`
4. `fft`
5. `glcm`
6. `lbp`
7. `noise`
8. `perturb`
9. `sim_camera`
10. `awb`
11. `lut`

## API Shape

Base path: `/api/v1`

Endpoints:

- `GET /health`
- `POST /api/v1/process/blend`
- `POST /api/v1/process/non-semantic`
- `POST /api/v1/process/clahe`
- `POST /api/v1/process/fft`
- `POST /api/v1/process/glcm`
- `POST /api/v1/process/lbp`
- `POST /api/v1/process/noise`
- `POST /api/v1/process/perturb`
- `POST /api/v1/process/sim-camera`
- `POST /api/v1/process/awb`
- `POST /api/v1/process/lut`
- `POST /api/v1/process/pipeline`

All processing endpoints return an image file in the response body.

## Transport Contract

All `POST` endpoints accept `multipart/form-data`.

Common fields:

- `image`: required uploaded input image.
- `output_format`: optional string. Allowed values: `png`, `jpeg`. Default: `png`.
- `include_exif`: optional boolean. Default: `true`.

Optional uploaded support files:

- `ref_image`: optional AWB reference image.
- `fft_ref_image`: optional FFT/GLCM/LBP reference image.
- `lut_file`: optional LUT file for LUT processing.

Stage configuration fields are sent as a JSON string in a `config` form field.

`config` rules:

- For single-stage endpoints, `config` contains only that stage's parameters.
- For the pipeline endpoint, `config` contains `stages`, `execution_order`, and `stage_order`.

## Single-Stage Contracts

### Blend

Endpoint: `POST /api/v1/process/blend`

`config` payload:

```json
{
  "blend_tolerance": 10.0,
  "blend_min_region": 50,
  "blend_max_samples": 100000,
  "blend_n_jobs": null
}
```

### Non-Semantic

Endpoint: `POST /api/v1/process/non-semantic`

`config` payload:

```json
{
  "ns_iterations": 500,
  "ns_learning_rate": 0.0003,
  "ns_t_lpips": 0.04,
  "ns_t_l2": 0.00003,
  "ns_c_lpips": 0.01,
  "ns_c_l2": 0.6,
  "ns_grad_clip": 0.05
}
```

### CLAHE

Endpoint: `POST /api/v1/process/clahe`

`config` payload:

```json
{
  "clahe_clip": 2.0,
  "tile": 8
}
```

### FFT

Endpoint: `POST /api/v1/process/fft`

`config` payload:

```json
{
  "fft_mode": "auto",
  "fft_alpha": 1.0,
  "cutoff": 0.25,
  "fstrength": 0.9,
  "randomness": 0.05,
  "phase_perturb": 0.08,
  "radial_smooth": 5,
  "fft_variant": "v2",
  "seed": null
}
```

Requires `fft_ref_image` only when the caller wants reference-driven matching.

### GLCM

Endpoint: `POST /api/v1/process/glcm`

`config` payload:

```json
{
  "glcm_distances": [1],
  "glcm_angles": [0.0, 0.7853981634, 1.5707963268, 2.3561944902],
  "glcm_levels": 256,
  "glcm_strength": 0.9,
  "seed": null
}
```

`fft_ref_image` is optional and reused as the texture reference.

### LBP

Endpoint: `POST /api/v1/process/lbp`

`config` payload:

```json
{
  "lbp_radius": 3,
  "lbp_n_points": 24,
  "lbp_method": "uniform",
  "lbp_strength": 0.9,
  "seed": null
}
```

`fft_ref_image` is optional and reused as the texture reference.

### Noise

Endpoint: `POST /api/v1/process/noise`

`config` payload:

```json
{
  "noise_std": 0.02,
  "seed": null
}
```

### Perturb

Endpoint: `POST /api/v1/process/perturb`

`config` payload:

```json
{
  "perturb_magnitude": 0.008,
  "seed": null
}
```

### Sim Camera

Endpoint: `POST /api/v1/process/sim-camera`

`config` payload:

```json
{
  "no_no_bayer": true,
  "jpeg_cycles": 1,
  "jpeg_qmin": 88,
  "jpeg_qmax": 96,
  "vignette_strength": 0.35,
  "chroma_strength": 1.2,
  "iso_scale": 1.0,
  "read_noise": 2.0,
  "hot_pixel_prob": 0.000001,
  "banding_strength": 0.0,
  "motion_blur_kernel": 1,
  "seed": null
}
```

### AWB

Endpoint: `POST /api/v1/process/awb`

`config` payload:

```json
{
  "seed": null
}
```

If `ref_image` is provided, AWB uses that reference. Otherwise it uses grey-world behavior.

### LUT

Endpoint: `POST /api/v1/process/lut`

`config` payload:

```json
{
  "lut_strength": 0.1
}
```

Requires `lut_file`.

## Pipeline Contract

Endpoint: `POST /api/v1/process/pipeline`

`config` payload:

```json
{
  "execution_order": false,
  "stage_order": ["noise", "clahe", "fft"],
  "stages": {
    "noise": {
      "noise_std": 0.02,
      "seed": 123
    },
    "clahe": {
      "clahe_clip": 2.0,
      "tile": 8
    },
    "fft": {
      "fft_mode": "auto",
      "fft_alpha": 1.0,
      "cutoff": 0.25,
      "fstrength": 0.9,
      "randomness": 0.05,
      "phase_perturb": 0.08,
      "radial_smooth": 5,
      "fft_variant": "v2",
      "seed": 123
    }
  }
}
```

Pipeline rules:

- `stages` is a map of enabled stage names to stage-specific config.
- If `execution_order` is `false` or omitted, execution uses the default stage order.
- If `execution_order` is `true`, execution uses `stage_order`.
- `stage_order` must contain each enabled stage at most once.
- Any enabled stage omitted from `stage_order` is invalid when `execution_order` is `true`.
- Any stage name in `stage_order` not present in `stages` is invalid.

## Response Contract

Successful responses:

- Status: `200 OK`
- Content type: `image/png` or `image/jpeg`
- Headers:
  - `X-Stage-Count`: number of stages executed
  - `X-Stage-Order`: comma-separated executed stage order

Error responses:

- Status: `400` for malformed config, missing required support files, unsupported stage names, or invalid stage ordering.
- Status: `422` for invalid request schema.
- Status: `500` for unexpected processing failures.

Error body:

```json
{
  "detail": "Human-readable error message"
}
```

## Internal Implementation Constraints

- The API must call a shared in-memory processing service rather than shelling out to the CLI.
- The CLI path in `image_postprocess/processor.py` must remain supported.
- Shared logic must keep the current default order unchanged.
- Shared logic must support explicit stage lists for API-driven execution order.