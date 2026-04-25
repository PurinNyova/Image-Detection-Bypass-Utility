# FastAPI Backend Contract

This document is the frontend-facing contract for the FastAPI service in this repository.
It is written so a separate frontend or agent can integrate with the API without reading backend code.

## Purpose

The API accepts an input image, applies one or more image-processing stages, and returns the processed image as a binary file response.

The API supports two integration patterns:

1. Single-stage processing: one endpoint per stage.
2. Pipeline processing: one endpoint that runs multiple stages in a single request.

## Server Defaults

- Local development host: `http://127.0.0.1:8000`
- OpenAPI UI when running locally: `http://127.0.0.1:8000/docs`
- API version prefix for processing routes: `/api/v1`

## Endpoint Inventory

Health endpoint:

- `GET /health`

Single-stage processing endpoints:

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

Multi-stage processing endpoint:

- `POST /api/v1/process/pipeline`

## Processing Model

The system recognizes these internal stage names:

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

Important naming rule:

- Route paths use kebab-case where applicable, for example `non-semantic` and `sim-camera`.
- JSON config payloads use snake_case stage names, for example `non_semantic` and `sim_camera`.

Default stage execution order:

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

## Transport Contract

All processing endpoints use `multipart/form-data`.

Every processing request sends form fields, not a raw JSON body.

### Common Form Fields

These fields are accepted by every processing endpoint.

`image`

- Type: file upload
- Required: yes
- Meaning: source image to process
- Accepted values: any image file that Pillow can decode
- Failure cases:
  - missing file -> HTTP `422`
  - empty file -> HTTP `400` with `detail: "image is empty"`
  - invalid image bytes -> HTTP `400` with `detail: "image is not a valid image"`

`config`

- Type: string
- Encoding: JSON serialized as a text form field
- Required:
  - single-stage endpoints: no, defaults to `{}`
  - pipeline endpoint: yes
- Meaning:
  - single-stage endpoints: stage-specific options object
  - pipeline endpoint: full pipeline configuration object
- Failure cases:
  - malformed JSON -> HTTP `400` with `detail` string starting with `Invalid JSON in config:`
  - schema mismatch -> HTTP `422` with structured validation errors

`output_format`

- Type: string
- Required: no
- Allowed values: `png`, `jpeg`
- Default: `png`
- Meaning: output image encoding used for the response body

`include_exif`

- Type: boolean-like form value
- Required: no
- Default: `true`
- Meaning: when `true`, generated EXIF metadata is embedded in the returned image when possible

`ref_image`

- Type: file upload
- Required: no
- Used by: `awb`
- Ignored by: all other stages
- Failure cases when supplied:
  - empty file -> HTTP `400` with `detail: "ref_image is empty"`
  - invalid image bytes -> HTTP `400` with `detail: "ref_image is not a valid image"`

`fft_ref_image`

- Type: file upload
- Required: no
- Used by: `fft`, `glcm`, `lbp`
- Ignored by: all other stages
- Failure cases when supplied:
  - empty file -> HTTP `400` with `detail: "fft_ref_image is empty"`
  - invalid image bytes -> HTTP `400` with `detail: "fft_ref_image is not a valid image"`

`lut_file`

- Type: file upload
- Required:
  - `POST /api/v1/process/lut`: yes
  - `POST /api/v1/process/pipeline`: yes if `lut` is enabled in `config.stages`
  - all other processing requests: optional and ignored
- Meaning: LUT definition file loaded and applied by the LUT stage
- Supported formats:
  - `.cube`
  - `.npy`
  - image-based LUT formats accepted by the LUT loader
- Failure cases when supplied:
  - empty file -> HTTP `400` with `detail: "lut_file is empty"`
  - unsupported or invalid LUT -> HTTP `400` with loader-provided error text

## Response Contract

### Success Response

All successful processing endpoints return the processed image as the raw response body.

- Status: `200 OK`
- Body: binary image bytes
- Content-Type:
  - `image/png` when `output_format=png`
  - `image/jpeg` when `output_format=jpeg`

Response headers:

- `X-Stage-Count`: number of stages executed
- `X-Stage-Order`: comma-separated stage names in execution order

Examples:

- Single-stage FFT request returns:
  - `X-Stage-Count: 1`
  - `X-Stage-Order: fft`
- Pipeline request running `noise`, `clahe`, `fft` returns:
  - `X-Stage-Count: 3`
  - `X-Stage-Order: noise,clahe,fft`

Frontend handling requirements:

1. Treat the response as a binary blob, not JSON.
2. Read `Content-Type` to determine preview or download handling.
3. Read `X-Stage-Count` and `X-Stage-Order` for confirmation UI, audit logs, or debugging.

### Health Response

`GET /health`

- Status: `200 OK`
- Content-Type: `application/json`
- Body:

```json
{
  "status": "ok"
}
```

### Error Responses

The backend emits three practical error shapes.

#### 400 Bad Request

Used for malformed JSON, empty uploads, invalid image bytes, missing required support files, invalid pipeline ordering, and unsupported FFT variants.

Shape:

```json
{
  "detail": "Human-readable error message"
}
```

Representative examples:

```json
{
  "detail": "Invalid JSON in config: Expecting ',' delimiter"
}
```

```json
{
  "detail": "lut_file is required for LUT processing"
}
```

```json
{
  "detail": "stage_order is required when execution_order is true"
}
```

```json
{
  "detail": "Unsupported fft_variant 'foo'"
}
```

#### 422 Unprocessable Entity

Used when request validation fails.

There are two common sources:

1. FastAPI request validation, for example a missing required `image` field.
2. Pydantic validation of the parsed `config` JSON.

Shape:

```json
{
  "detail": [
    {
      "type": "validation_error_type",
      "loc": ["field", "path"],
      "msg": "Human-readable explanation",
      "input": "offending value"
    }
  ]
}
```

Representative example:

```json
{
  "detail": [
    {
      "type": "extra_forbidden",
      "loc": ["blend_extra_field"],
      "msg": "Extra inputs are not permitted",
      "input": 123
    }
  ]
}
```

Validation rule to note:

- Stage config models are strict. Unknown fields are rejected.

#### 500 Internal Server Error

Used when processing raises an unhandled exception that the API does not convert into a `400` or `422`.

Shape:

```json
{
  "detail": "Runtime exception message"
}
```

Frontend expectation:

- Treat `500` as a processing failure and surface the `detail` string to the user or logs.

## Single-Stage Endpoint Pattern

All single-stage endpoints behave the same way:

1. Parse `config` JSON into that stage's schema.
2. Load the main `image`.
3. Optionally load `ref_image`, `fft_ref_image`, and `lut_file`.
4. Run exactly one stage.
5. Return the processed image.

For single-stage endpoints, omitting `config` is valid and uses defaults.

## Pipeline Endpoint Pattern

The pipeline endpoint allows multiple stages in one request.

It validates:

1. `config.stages` exists and is not empty.
2. Every stage key is a supported stage name.
3. Each stage config conforms exactly to that stage schema.
4. If `execution_order=true`, `stage_order` must be present, contain no duplicates, contain all enabled stages, and contain no extra stages.

## Stage Dependency Matrix

This matrix shows which support files matter to which stage.

- `blend`: only `image`
- `non_semantic`: only `image`
- `clahe`: only `image`
- `fft`: `image`, optional `fft_ref_image`
- `glcm`: `image`, optional `fft_ref_image`
- `lbp`: `image`, optional `fft_ref_image`
- `noise`: only `image`
- `perturb`: only `image`
- `sim_camera`: only `image`
- `awb`: `image`, optional `ref_image`
- `lut`: `image`, required `lut_file`

Behavior notes:

- `awb` without `ref_image` falls back to grey-world balancing.
- `fft` supports both reference-driven and model-driven operation.
- `glcm` and `lbp` can use `fft_ref_image` as their texture reference.
- `lut` always requires `lut_file`.

## FFT Variant Rules

Allowed `fft_variant` values:

- `v1 (Original)`
- `v2`
- `v3`

Notes:

- Default is `v2`.
- If `fft_variant` is not one of the values above, the API returns HTTP `400`.
- `phase_perturb` is effectively ignored by the `v3` implementation.

## Request Examples

### Health Check

```bash
curl http://127.0.0.1:8000/health
```

### Single-Stage Request Example

Noise stage using only defaults except for output format:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/process/noise \
  -F "image=@sample.png" \
  --form-string 'config={"noise_std":0.02,"seed":123}' \
  -F "output_format=png" \
  -F "include_exif=true" \
  -o output.png
```

### FFT Request Example With Reference Image

```bash
curl -X POST http://127.0.0.1:8000/api/v1/process/fft \
  -F "image=@sample.png" \
  -F "fft_ref_image=@fft_ref.png" \
  --form-string 'config={"fft_mode":"auto","fft_alpha":1.0,"cutoff":0.25,"fstrength":0.9,"randomness":0.05,"phase_perturb":0.08,"radial_smooth":5,"fft_variant":"v2","seed":123}' \
  -F "output_format=jpeg" \
  -F "include_exif=true" \
  -o output.jpg
```

### Pipeline Request Example

```bash
curl -X POST http://127.0.0.1:8000/api/v1/process/pipeline \
  -F "image=@sample.png" \
  -F "fft_ref_image=@fft_ref.png" \
  --form-string 'config={"execution_order":true,"stage_order":["noise","clahe","fft"],"stages":{"noise":{"noise_std":0.02,"seed":123},"clahe":{"clahe_clip":2.0,"tile":8},"fft":{"fft_mode":"auto","fft_alpha":1.0,"cutoff":0.25,"fstrength":0.9,"randomness":0.05,"phase_perturb":0.08,"radial_smooth":5,"fft_variant":"v2","seed":123}}}' \
  -F "output_format=png" \
  -F "include_exif=true" \
  -o output.png
```

### Frontend Fetch Example

```js
const formData = new FormData();
formData.append('image', fileInput.files[0]);
formData.append('config', JSON.stringify({
  noise_std: 0.02,
  seed: 123,
}));
formData.append('output_format', 'png');
formData.append('include_exif', 'true');

const response = await fetch('http://127.0.0.1:8000/api/v1/process/noise', {
  method: 'POST',
  body: formData,
});

if (!response.ok) {
  const errorPayload = await response.json();
  throw new Error(errorPayload.detail || 'Request failed');
}

const contentType = response.headers.get('Content-Type');
const stageOrder = response.headers.get('X-Stage-Order');
const blob = await response.blob();
const objectUrl = URL.createObjectURL(blob);

console.log({ contentType, stageOrder, objectUrl });
```

## Endpoint Specifications

The sections below define every request contract in detail.

---

## `GET /health`

Purpose:

- Liveness check for the API process.

Request body:

- none

Success response:

```json
{
  "status": "ok"
}
```

---

## `POST /api/v1/process/blend`

Purpose:

- Reduces or blends color regions using tolerance and region controls.

Accepted support files:

- `image` required
- `ref_image` accepted but ignored
- `fft_ref_image` accepted but ignored
- `lut_file` accepted but ignored

`config` schema:

```json
{
  "blend_tolerance": 10.0,
  "blend_min_region": 50,
  "blend_max_samples": 100000,
  "blend_n_jobs": null
}
```

Field definitions:

- `blend_tolerance`: number, default `10.0`, color distance threshold
- `blend_min_region`: integer, default `50`, minimum retained region size
- `blend_max_samples`: integer, default `100000`, maximum sampled pixels for clustering
- `blend_n_jobs`: integer or `null`, default `null`, optional worker count

Success headers:

- `X-Stage-Count: 1`
- `X-Stage-Order: blend`

---

## `POST /api/v1/process/non-semantic`

Purpose:

- Runs the non-semantic attack path using LPIPS- and L2-based optimization.

Accepted support files:

- `image` required
- all support uploads otherwise ignored

`config` schema:

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

Field definitions:

- `ns_iterations`: integer, default `500`, optimization step count
- `ns_learning_rate`: number, default `0.0003`, optimizer step size
- `ns_t_lpips`: number, default `0.04`, LPIPS threshold
- `ns_t_l2`: number, default `0.00003`, L2 threshold
- `ns_c_lpips`: number, default `0.01`, LPIPS penalty weight
- `ns_c_l2`: number, default `0.6`, L2 penalty weight
- `ns_grad_clip`: number, default `0.05`, gradient clipping threshold

Success headers:

- `X-Stage-Count: 1`
- `X-Stage-Order: non_semantic`

Operational note:

- Runtime availability depends on the optional non-semantic processing stack being installed and functioning.

---

## `POST /api/v1/process/clahe`

Purpose:

- Applies CLAHE-based color correction.

Accepted support files:

- `image` required
- all support uploads otherwise ignored

`config` schema:

```json
{
  "clahe_clip": 2.0,
  "tile": 8
}
```

Field definitions:

- `clahe_clip`: number, default `2.0`, clip limit
- `tile`: integer, default `8`, tile grid width and height

Success headers:

- `X-Stage-Count: 1`
- `X-Stage-Order: clahe`

---

## `POST /api/v1/process/fft`

Purpose:

- Applies spectral matching in the Fourier domain.

Accepted support files:

- `image` required
- `fft_ref_image` optional, used when reference-driven matching is desired
- `ref_image` accepted but ignored
- `lut_file` accepted but ignored

`config` schema:

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

Field definitions:

- `fft_mode`: string enum, default `auto`, one of `auto`, `ref`, `model`
- `fft_alpha`: number, default `1.0`, slope used by model mode
- `cutoff`: number, default `0.25`, low-frequency cutoff
- `fstrength`: number, default `0.9`, blend strength
- `randomness`: number, default `0.05`, stochastic modulation amount
- `phase_perturb`: number, default `0.08`, phase perturbation in radians
- `radial_smooth`: integer, default `5`, radial smoothing bins
- `fft_variant`: string, default `v2`, allowed values `v1 (Original)`, `v2`, `v3`
- `seed`: integer or `null`, default `null`, reproducibility control

Behavior notes:

- `fft_mode=auto` uses the most appropriate mode based on whether a reference image is present.
- `fft_mode=ref` is meaningful only when `fft_ref_image` is supplied.
- `fft_variant=v3` ignores `phase_perturb`.

Success headers:

- `X-Stage-Count: 1`
- `X-Stage-Order: fft`

---

## `POST /api/v1/process/glcm`

Purpose:

- Matches luminance texture statistics using a gray-level co-occurrence matrix.

Accepted support files:

- `image` required
- `fft_ref_image` optional texture reference
- other support uploads ignored

`config` schema:

```json
{
  "glcm_distances": [1],
  "glcm_angles": [0.0, 0.7853981634, 1.5707963268, 2.3561944902],
  "glcm_levels": 256,
  "glcm_strength": 0.9,
  "seed": null
}
```

Field definitions:

- `glcm_distances`: array of integers, default `[1]`, neighbor distances in pixels
- `glcm_angles`: array of numbers, default `[0.0, 0.7853981634, 1.5707963268, 2.3561944902]`, angles in radians
- `glcm_levels`: integer, default `256`, quantized gray levels
- `glcm_strength`: number, default `0.9`, blend strength
- `seed`: integer or `null`, default `null`, reproducibility control

Success headers:

- `X-Stage-Count: 1`
- `X-Stage-Order: glcm`

---

## `POST /api/v1/process/lbp`

Purpose:

- Matches texture characteristics using local binary patterns.

Accepted support files:

- `image` required
- `fft_ref_image` optional texture reference
- other support uploads ignored

`config` schema:

```json
{
  "lbp_radius": 3,
  "lbp_n_points": 24,
  "lbp_method": "uniform",
  "lbp_strength": 0.9,
  "seed": null
}
```

Field definitions:

- `lbp_radius`: integer, default `3`, LBP radius
- `lbp_n_points`: integer, default `24`, circular neighbor count
- `lbp_method`: string enum, default `uniform`, one of `default`, `ror`, `uniform`, `var`
- `lbp_strength`: number, default `0.9`, blend strength
- `seed`: integer or `null`, default `null`, reproducibility control

Success headers:

- `X-Stage-Count: 1`
- `X-Stage-Order: lbp`

---

## `POST /api/v1/process/noise`

Purpose:

- Adds Gaussian noise.

Accepted support files:

- `image` required
- other support uploads ignored

`config` schema:

```json
{
  "noise_std": 0.02,
  "seed": null
}
```

Field definitions:

- `noise_std`: number, default `0.02`, standard deviation as a fraction of 255
- `seed`: integer or `null`, default `null`, reproducibility control

Success headers:

- `X-Stage-Count: 1`
- `X-Stage-Order: noise`

---

## `POST /api/v1/process/perturb`

Purpose:

- Applies randomized pixel-space perturbations.

Accepted support files:

- `image` required
- other support uploads ignored

`config` schema:

```json
{
  "perturb_magnitude": 0.008,
  "seed": null
}
```

Field definitions:

- `perturb_magnitude`: number, default `0.008`, perturbation magnitude fraction
- `seed`: integer or `null`, default `null`, reproducibility control

Success headers:

- `X-Stage-Count: 1`
- `X-Stage-Order: perturb`

---

## `POST /api/v1/process/sim-camera`

Purpose:

- Simulates camera pipeline artifacts such as Bayer behavior, JPEG cycles, vignette, chromatic aberration, sensor noise, and motion blur.

Accepted support files:

- `image` required
- other support uploads ignored

`config` schema:

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

Field definitions:

- `no_no_bayer`: boolean, default `true`, legacy double-negative flag; `true` disables Bayer or demosaic simulation, `false` enables it
- `jpeg_cycles`: integer, default `1`, number of JPEG recompression passes
- `jpeg_qmin`: integer, default `88`, minimum JPEG quality
- `jpeg_qmax`: integer, default `96`, maximum JPEG quality
- `vignette_strength`: number, default `0.35`, vignette intensity
- `chroma_strength`: number, default `1.2`, chromatic aberration strength
- `iso_scale`: number, default `1.0`, exposure or ISO scaling factor for noise simulation
- `read_noise`: number, default `2.0`, read-noise sigma
- `hot_pixel_prob`: number, default `0.000001`, probability of hot pixels
- `banding_strength`: number, default `0.0`, horizontal banding amount
- `motion_blur_kernel`: integer, default `1`, motion blur kernel size where `1` means effectively off
- `seed`: integer or `null`, default `null`, reproducibility control

Success headers:

- `X-Stage-Count: 1`
- `X-Stage-Order: sim_camera`

---

## `POST /api/v1/process/awb`

Purpose:

- Applies auto white balance.

Accepted support files:

- `image` required
- `ref_image` optional AWB reference
- `fft_ref_image` accepted but ignored
- `lut_file` accepted but ignored

`config` schema:

```json
{
  "seed": null
}
```

Field definitions:

- `seed`: integer or `null`, default `null`, currently accepted for schema consistency

Behavior notes:

- When `ref_image` is present, AWB uses it as the white-balance reference.
- When `ref_image` is omitted, AWB uses grey-world behavior.

Success headers:

- `X-Stage-Count: 1`
- `X-Stage-Order: awb`

---

## `POST /api/v1/process/lut`

Purpose:

- Applies a LUT after all other enabled processing for that request.

Accepted support files:

- `image` required
- `lut_file` required
- other support uploads ignored

`config` schema:

```json
{
  "lut_strength": 0.1
}
```

Field definitions:

- `lut_strength`: number, default `0.1`, blend amount from `0.0` to full LUT effect

Failure rules:

- Missing `lut_file` -> HTTP `400`

Success headers:

- `X-Stage-Count: 1`
- `X-Stage-Order: lut`

---

## `POST /api/v1/process/pipeline`

Purpose:

- Runs multiple stages in one request.

Accepted support files:

- `image` required
- `ref_image` optional, used only if `awb` is enabled in `config.stages`
- `fft_ref_image` optional, used only if `fft`, `glcm`, or `lbp` is enabled in `config.stages`
- `lut_file` required only if `lut` is enabled in `config.stages`

`config` schema:

```json
{
  "execution_order": false,
  "stage_order": [],
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

Top-level field definitions:

- `execution_order`: boolean, default `false`
- `stage_order`: array of stage names, default `[]`
- `stages`: object map from stage name to that stage's config object, default `{}`

Pipeline validation rules:

1. `stages` must contain at least one stage.
2. Every key inside `stages` must be one of the supported stage names.
3. Every stage payload must satisfy that stage's strict schema.
4. If `execution_order=false`, `stage_order` is ignored and execution follows the default order filtered to enabled stages.
5. If `execution_order=true`, `stage_order` is mandatory.
6. If `execution_order=true`, `stage_order` cannot contain duplicates.
7. If `execution_order=true`, `stage_order` must include every enabled stage exactly once.
8. If `execution_order=true`, `stage_order` cannot contain stages that are not present in `stages`.

Representative invalid pipeline examples:

Missing `stage_order` while `execution_order=true`:

```json
{
  "detail": "stage_order is required when execution_order is true"
}
```

Duplicate stage names:

```json
{
  "detail": "stage_order cannot contain duplicates"
}
```

Missing enabled stage from `stage_order`:

```json
{
  "detail": "stage_order is missing enabled stages: fft"
}
```

Extra stage in `stage_order`:

```json
{
  "detail": "stage_order contains stages not present in stages: awb"
}
```

Pipeline success behavior:

- The response body is the final processed image after all enabled stages run.
- `X-Stage-Count` equals the number of enabled stages.
- `X-Stage-Order` reports the actual execution order.

## Stage Config Reuse Rules

For pipeline requests, each entry inside `config.stages` reuses the same schema as the corresponding single-stage endpoint.

That means:

- the payload under `stages.noise` must match the `noise` config schema
- the payload under `stages.fft` must match the `fft` config schema
- the payload under `stages.lut` must match the `lut` config schema

Unknown fields are rejected everywhere.

## Frontend Integration Guidance

### Building Requests

1. Always send `multipart/form-data`.
2. Serialize `config` with `JSON.stringify(...)` before appending it to `FormData`.
3. Send booleans as strings, for example `true` or `false`.
4. Do not send unknown fields in any stage config.
5. For the pipeline endpoint, send only enabled stages inside `config.stages`.

### Handling Responses

1. On success, read the response as a blob.
2. Use `Content-Type` to decide whether to preview or download.
3. Use `X-Stage-Order` and `X-Stage-Count` for user-visible execution summaries.
4. On failure, first try to parse JSON and display `detail`.

### Recommended UI Constraints

To avoid preventable server-side validation failures:

- restrict `output_format` to `png` or `jpeg`
- restrict `fft_mode` to `auto`, `ref`, or `model`
- restrict `lbp_method` to `default`, `ror`, `uniform`, or `var`
- restrict `fft_variant` to `v1 (Original)`, `v2`, or `v3`
- require a LUT file before allowing LUT submission
- require `stage_order` when pipeline `execution_order` is enabled

## Minimal JSON Schemas By Endpoint

These are the exact config defaults a client can use as initial form state.

`blend`

```json
{
  "blend_tolerance": 10.0,
  "blend_min_region": 50,
  "blend_max_samples": 100000,
  "blend_n_jobs": null
}
```

`non_semantic`

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

`clahe`

```json
{
  "clahe_clip": 2.0,
  "tile": 8
}
```

`fft`

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

`glcm`

```json
{
  "glcm_distances": [1],
  "glcm_angles": [0.0, 0.7853981634, 1.5707963268, 2.3561944902],
  "glcm_levels": 256,
  "glcm_strength": 0.9,
  "seed": null
}
```

`lbp`

```json
{
  "lbp_radius": 3,
  "lbp_n_points": 24,
  "lbp_method": "uniform",
  "lbp_strength": 0.9,
  "seed": null
}
```

`noise`

```json
{
  "noise_std": 0.02,
  "seed": null
}
```

`perturb`

```json
{
  "perturb_magnitude": 0.008,
  "seed": null
}
```

`sim_camera`

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

`awb`

```json
{
  "seed": null
}
```

`lut`

```json
{
  "lut_strength": 0.1
}
```

`pipeline`

```json
{
  "execution_order": false,
  "stage_order": [],
  "stages": {}
}
```

## Summary For Frontend Authors

If you only need the minimum viable integration behavior, it is this:

1. Send `multipart/form-data`.
2. Always include `image`.
3. Include `config` as a JSON string.
4. For LUT requests, include `lut_file`.
5. For AWB requests, optionally include `ref_image`.
6. For FFT, GLCM, or LBP requests, optionally include `fft_ref_image`.
7. Read successful responses as image blobs.
8. Read error responses as JSON with a `detail` field.