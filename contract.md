# FastAPI Backend Contract

This document is the frontend-facing contract for the FastAPI service in this
repository (`api_backend`). It is written so a separate frontend or agent can
integrate with the API without reading backend code.

## Purpose

The API accepts one input image per request, applies exactly one image
process, and returns the processed image as raw binary bytes.

There is no pipeline endpoint and no server-selected execution order. Order
belongs entirely to the client: every successful response can be sent as the
`image` field of any other endpoint. The server never sees, stores, or chooses
a stage sequence.

## Server Defaults

- Local development host: `http://127.0.0.1:8000`
- OpenAPI UI when running locally: `http://127.0.0.1:8000/docs`
- API version prefix for processing routes: `/api/v1`

## Endpoint Inventory

Health:

- `GET /health`

Ordinary processes:

- `POST /api/v1/color-blend`
- `POST /api/v1/non-semantic`
- `POST /api/v1/clahe`
- `POST /api/v1/fft`
- `POST /api/v1/glcm`
- `POST /api/v1/lbp`
- `POST /api/v1/noise`
- `POST /api/v1/perturb`
- `POST /api/v1/sim-camera`
- `POST /api/v1/awb`
- `POST /api/v1/lut`

Finalizer:

- `POST /api/v1/forensic-camera`

There is no `POST /api/v1/process/pipeline`. There is no endpoint that accepts
or executes a server-ordered multi-stage sequence.

## Shared Multipart Contract

Every processing endpoint uses `multipart/form-data`. Form fields other than
the ones listed for a given endpoint are rejected with HTTP `422` (unknown
form field). This is why, for example, sending `reference_image` or `lut_file`
to `/noise` fails instead of being silently ignored.

Route-specific reference and LUT upload fields are named consistently:

- `reference_image`: the single public reference upload field (FFT, GLCM, LBP, AWB).
- `lut_file`: the single public LUT upload field (LUT only).

### Shared Fields (all processing endpoints)

`image`

- Type: file upload
- Required: yes
- Meaning: source image to process, produced by the client or by a previous
  endpoint response
- Accepted values: any image file Pillow can decode
- Failure cases:
  - missing file -> HTTP `422`
  - empty file -> HTTP `400` with `detail: "image is empty"`
  - invalid image bytes -> HTTP `400` with `detail: "image is not a valid image"`

`config`

- Type: string, JSON serialized as a text form field
- Required: no (defaults to `{}`, which means: every accepted field gets its default)
- Failure cases:
  - malformed JSON -> HTTP `400` with `detail` starting with `Invalid JSON in config:`
  - schema mismatch (including any unknown field) -> HTTP `422` with
    structured validation errors

`output_format`

- Type: string
- Required: no
- Allowed values: `png`, `jpeg`
- Default: `png`
- Meaning: output encoding for the response body
- Not accepted by `/forensic-camera` (that route alone always returns JPEG).

`include_exif`

- Type: boolean-like form value (send the string `true` or `false`)
- Default: **`false`**
- Meaning: opt-in legacy EXIF embedding for ordinary processes; applies only
  to ordinary endpoints and is meaningless on `/forensic-camera`, which
  writes its own full camera EXIF.

## Upload Size Limit

Every uploaded file (`image`, `reference_image`, `lut_file`) is capped at
64 MiB (67108864 bytes) per file. Uploads are read in bounded chunks; the
moment a file exceeds the cap processing stops and the request fails.

- Exceeding the cap -> HTTP `413` with `detail` like
  `"image exceeds 67108864 byte upload cap"`

## Shared Response Contract

Success (all ordinary endpoints):

- Status: `200 OK`
- Body: raw processed image bytes
- `Content-Type`: `image/png` or `image/jpeg`
- Headers:
  - `X-Process`: the route segment name, e.g. `noise`
  - `X-Output-Format`: `png` or `jpeg`

There are no stage-list headers. Do not look for `X-Stage-Count` or
`X-Stage-Order`; they are not emitted.

Every ordinary response must be accepted as the `image` field of every other
ordinary endpoint.

Client handling requirements:

1. Treat the response as a binary blob, not JSON.
2. Read `Content-Type` to decide preview vs download handling.
3. On error, try to parse the body as JSON and display `detail`.

## Health

`GET /health`

- Status: `200 OK`
- `Content-Type: application/json`
- Body: `{"status": "ok"}`

## Error Responses

### 400 Bad Request

Shape: `{"detail": "Human-readable message"}`

Used for:

- malformed JSON in `config` (`Invalid JSON in config: ...`)
- empty upload field (`"... is empty"`)
- invalid image OR LUT bytes (`"... is not a valid image"` / loader error text)
- missing required LUT upload (`"lut_file is required for LUT processing"`)
- unsupported or invalid LUT content (loader-provided message)

### 413 Payload Too Large

Shape: `{"detail": "<field> exceeds 67108864 byte upload cap"}`

Used when any single uploaded file exceeds 64 MiB.

### 422 Unprocessable Entity

Used for request validation failures:

- FastAPI validation (e.g. missing required `image`)
- Pydantic validation of the parsed `config` (strict models reject unknown
  fields and wrong types)
- unknown multipart form fields on a route
- invalid `datetime_original` on `/forensic-camera`

Shape (structured):

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

Shape (unknown form field / bad datetime, plain string):

```json
{"detail": "Unknown form field(s): reference_image"}
```

### 500 Internal Server Error

Shape: `{"detail": "Internal processing error"}`

Used when processing raises an unhandled exception. Raw exception details are
not returned to clients; the traceback is logged server-side.

## Composition Rules

- The order of calls belongs entirely to the client. Any ordinary response may
  be fed to any other ordinary endpoint as its `image`.
- `/forensic-camera` is a JPEG finalizer and is recommended as the LAST
  request in any composition. Sending forensic-camera output into another
  endpoint may re-encode it and destroy the final JPEG metadata and
  quantization characteristics. Clients are free to do so, but should not
  expect forensic metadata to survive.

## Reference Behavior Notes

- AWB without `reference_image` falls back to grey-world balancing.
- FFT supports reference-driven (`reference_image` supplied) and model-driven
  operation.
- GLCM and LBP use `reference_image` as their optional texture reference; both
  also operate without one.
- LUT always requires `lut_file`; without it the request is a client error.
- `/forensic-camera` accepts only `image` and `config`; it does NOT accept
  `reference_image`, `lut_file`, `output_format`, or `include_exif`.

## Config Schemas (Strict)

All config models reject unknown fields (`422`). Each config matches one
endpoint; there is no shared numeric stage schema. Defaults below are what the
server applies when the equivalent field is omitted.

### `POST /api/v1/color-blend`

```json
{
  "blend_tolerance": 10.0,
  "blend_min_region": 50,
  "blend_max_samples": 100000,
  "blend_n_jobs": null
}
```

- `blend_tolerance`: number, default `10.0`, color distance threshold
- `blend_min_region`: integer, default `50`, minimum retained region size
- `blend_max_samples`: integer, default `100000`, max sampled pixels used in clustering
- `blend_n_jobs`: integer or `null`, default `null`, worker count override

### `POST /api/v1/non-semantic`

```json
{
  "ns_iterations": 500,
  "ns_learning_rate": 0.0003,
  "ns_t_lpips": 0.04,
  "ns_t_l2": 0.00003,
  "ns_c_lpips": 0.01,
  "ns_c_l2": 0.6,
  "ns_grad_clip": 0.05,
  "ns_adaptive_c_lpips": true,
  "ns_c_lpips_min": 0.0001,
  "ns_c_lpips_max": 1.0,
  "ns_search_interval": 40
}
```

- `ns_iterations`: integer, default `500`, optimization step count
- `ns_learning_rate`: number, default `0.0003`, optimizer step size
- `ns_t_lpips`: number, default `0.04`, LPIPS threshold
- `ns_t_l2`: number, default `0.00003`, L2 threshold
- `ns_c_lpips`: number, default `0.01`, LPIPS penalty weight
- `ns_c_l2`: number, default `0.6`, L2 penalty weight
- `ns_grad_clip`: number, default `0.05`, gradient clipping threshold
- `ns_adaptive_c_lpips`: boolean, default `true`, enables adaptive LPIPS weight search
- `ns_c_lpips_min`: number, default `0.0001`, lower bound of adaptive LPIPS weight search
- `ns_c_lpips_max`: number, default `1.0`, upper bound of adaptive LPIPS weight search
- `ns_search_interval`: integer, default `40`, how often the adaptive weight is re-evaluated in steps

Operational note: runtime availability depends on the optional non-semantic
processing stack being installed and working.

### `POST /api/v1/clahe`

```json
{
  "clahe_clip": 2.0,
  "tile": 8
}
```

- `clahe_clip`: number, default `2.0`, clip limit
- `tile`: integer, default `8`, tile grid width and height

### `POST /api/v1/fft`

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

- `fft_mode`: string, default `"auto"`, one of `"auto"`, `"ref"`, `"model"`
- `fft_alpha`: number, default `1.0`, slope used by model mode
- `cutoff`: number, default `0.25`, low-frequency cutoff
- `fstrength`: number, default `0.9`, blend strength
- `randomness`: number, default `0.05`, stochastic modulation amount
- `phase_perturb`: number, default `0.08`, phase perturbation in radians (ignored by `v3` and `v4`)
- `radial_smooth`: integer, default `5`, radial smoothing bins
- `fft_variant`: string, default `"v2"`, one of `"v1 (Original)"`, `"v2"`, `"v3"`, `"v4"`
- `seed`: integer or `null`, default `null`, reproducibility control

Notes:

- `fft_mode=auto` picks reference-driven matching when `reference_image` is supplied, otherwise model mode.
- `fft_mode=ref` is meaningful only when `reference_image` is supplied.

### `POST /api/v1/glcm`

```json
{
  "glcm_distances": [1],
  "glcm_angles": [0.0, 0.7853981634, 1.5707963268, 2.3561944902],
  "glcm_levels": 256,
  "glcm_strength": 0.9,
  "seed": null
}
```

- `glcm_distances`: array of integers, default `[1]`, neighbor distances in pixels
- `glcm_angles`: array of numbers, default `[0.0, 0.7853981634, 1.5707963268, 2.3561944902]`, angles in radians
- `glcm_levels`: integer, default `256`, quantized gray levels
- `glcm_strength`: number, default `0.9`, blend strength
- `seed`: integer or `null`, default `null`, reproducibility control

### `POST /api/v1/lbp`

```json
{
  "lbp_radius": 3,
  "lbp_n_points": 24,
  "lbp_method": "uniform",
  "lbp_strength": 0.9,
  "seed": null
}
```

- `lbp_radius`: integer, default `3`, LBP radius
- `lbp_n_points`: integer, default `24`, circular neighbor count
- `lbp_method`: string, default `"uniform"`, one of `"default"`, `"ror"`, `"uniform"`, `"var"`
- `lbp_strength`: number, default `0.9`, blend strength
- `seed`: integer or `null`, default `null`, reproducibility control

### `POST /api/v1/noise`

```json
{
  "noise_std": 0.02,
  "seed": null
}
```

- `noise_std`: number, default `0.02`, Gaussian sigma as a fraction of 255
- `seed`: integer or `null`, default `null`, reproducibility control

### `POST /api/v1/perturb`

```json
{
  "perturb_magnitude": 0.008,
  "seed": null
}
```

- `perturb_magnitude`: number, default `0.008`, perturbation magnitude fraction
- `seed`: integer or `null`, default `null`, reproducibility control

### `POST /api/v1/sim-camera`

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

- `no_no_bayer`: boolean, default `true`; legacy double-negative flag, `true` disables Bayer simulation, `false` enables it
- `jpeg_cycles`: integer, default `1`, JPEG recompression passes
- `jpeg_qmin`: integer, default `88`, minimum JPEG quality
- `jpeg_qmax`: integer, default `96`, maximum JPEG quality
- `vignette_strength`: number, default `0.35`, vignette intensity
- `chroma_strength`: number, default `1.2`, chromatic aberration strength
- `iso_scale`: number, default `1.0`, exposure/ISO scaling for noise simulation
- `read_noise`: number, default `2.0`, read-noise sigma
- `hot_pixel_prob`: number, default `0.000001`, hot-pixel probability
- `banding_strength`: number, default `0.0`, horizontal banding amount
- `motion_blur_kernel`: integer, default `1`, motion-blur kernel size, `1` = effectively off
- `seed`: integer or `null`, default `null`, reproducibility control

### `POST /api/v1/awb`

```json
{
  "seed": null
}
```

- `seed`: integer or `null`, default `null`, currently accepted for schema consistency

Behavior: uses `reference_image` when supplied, else grey-world fallback.

### `POST /api/v1/lut`

```json
{
  "lut_strength": 0.1
}
```

- `lut_strength`: number, default `0.1`, blend amount toward the full LUT effect

`lut_file` is required on this endpoint; missing -> HTTP `400`.

### `POST /api/v1/forensic-camera`

Finalizer. Accepts exactly `image` and `config` (no other form fields).

Always returns JPEG regardless of any other setting. Recommended last request.

```json
{
  "profile": "iphone_16_pro",
  "software": null,
  "datetime_original": null,
  "iso": null,
  "gps_lat": null,
  "gps_lon": null,
  "gps_alt": null,
  "ela_flatten": true,
  "strip_fingerprints": true,
  "seed": null
}
```

- `profile`: string, default `"iphone_16_pro"`; currently the only accepted value
- `software`: string or `null`, default `null`; Software EXIF value (server applies its own default, version-only string)
- `datetime_original`: string or `null`, default `null`; accepted formats `"YYYY-MM-DD HH:MM:SS"` or `"YYYY:MM:DD HH:MM:SS"`; other formats -> HTTP `422`
- `iso`: integer or `null`, default `null`
- `gps_lat`: number or `null`, default `null`
- `gps_lon`: number or `null`, default `null`
- `gps_alt`: number or `null`, default `null`
- `ela_flatten`: boolean, default `true`; whole-frame light flatten so error-level analysis looks like a camera JPEG
- `strip_fingerprints`: boolean, default `true`; strips JFIF/XMP and other telltale metadata
- `seed`: integer or `null`, default `null`, reproducibility control

Success response:

- Status: `200 OK`
- `Content-Type: image/jpeg`
- `X-Process: forensic-camera`
- `X-Output-Format: jpeg`
- `X-Forensic-Metadata`:
  - `full` when ExifTool is available (complete camera EXIF plus MakerNote)
  - `degraded` when ExifTool is unavailable (Pillow EXIF only, no MakerNote)

## Request Examples

### Health Check

```bash
curl http://127.0.0.1:8000/health
```

### Single-Process Request

```bash
curl -X POST http://127.0.0.1:8000/api/v1/noise \
  -F "image=@sample.png" \
  --form-string 'config={"noise_std":0.02,"seed":123}' \
  -F "output_format=png" \
  -F "include_exif=false" \
  -o noise.png
```

### Client-Selected Composition A (curl)

`clahe` -> `noise` -> `color-blend` -> `lut` -> `sim-camera` -> `forensic-camera`

Each response's bytes are forwarded as the next request's `image`, and
forensic-camera runs last:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/clahe \
  -F "image=@sample.png" --form-string 'config={"clahe_clip":2.0,"tile":8}' \
  -o step1.png

curl -X POST http://127.0.0.1:8000/api/v1/noise \
  -F "image=@step1.png" --form-string 'config={"noise_std":0.02,"seed":123}' \
  -o step2.png

curl -X POST http://127.0.0.1:8000/api/v1/color-blend \
  -F "image=@step2.png" --form-string 'config={"blend_tolerance":10.0}' \
  -o step3.png

curl -X POST http://127.0.0.1:8000/api/v1/lut \
  -F "image=@step3.png" --form-string 'config={"lut_strength":0.1}' \
  -F "lut_file=@grading.cube" \
  -o step4.png

curl -X POST http://127.0.0.1:8000/api/v1/sim-camera \
  -F "image=@step4.png" --form-string 'config={"jpeg_cycles":2,"seed":7}' \
  -F "output_format=png" \
  -o step5.png

curl -X POST http://127.0.0.1:8000/api/v1/forensic-camera \
  -F "image=@step5.png" --form-string 'config={"profile":"iphone_16_pro"}' \
  -o final.jpg
```

### Client-Selected Composition B (curl)

`perturb` -> `fft` (with reference) -> `awb` (no reference) -> `forensic-camera`

```bash
curl -X POST http://127.0.0.1:8000/api/v1/perturb \
  -F "image=@sample.png" --form-string 'config={"perturb_magnitude":0.008,"seed":123}' \
  -o step1.png

curl -X POST http://127.0.0.1:8000/api/v1/fft \
  -F "image=@step1.png" \
  -F "reference_image=@camera_ref.jpg" \
  --form-string 'config={"fft_mode":"ref","fft_variant":"v4","fstrength":0.9,"seed":123}' \
  -o step2.png

curl -X POST http://127.0.0.1:8000/api/v1/awb \
  -F "image=@step2.png" --form-string 'config={}' \
  -o step3.png

curl -X POST http://127.0.0.1:8000/api/v1/forensic-camera \
  -F "image=@step3.png" --form-string 'config={"datetime_original":"2026-09-19 10:30:00","iso":100}' \
  -o final.jpg
```

### JavaScript Composition (blob forwarding)

```js
const BASE = 'http://127.0.0.1:8000/api/v1';

async function run(endpoint, blob, config, extras = {}) {
  const formData = new FormData();
  const forensic = endpoint === 'forensic-camera'; // /forensic-camera accepts ONLY image and config
  formData.append('image', blob);
  formData.append('config', JSON.stringify(config));
  if (!forensic) {
    if (extras.reference_image) formData.append('reference_image', extras.reference_image);
    if (extras.lut_file) formData.append('lut_file', extras.lut_file);
    if (extras.output_format) formData.append('output_format', extras.output_format);
    formData.append('include_exif', 'false');
  }

  const response = await fetch(`${BASE}/${endpoint}`, { method: 'POST', body: formData });
  if (!response.ok) {
    const errorPayload = await response.json();
    throw new Error(errorPayload.detail || `Request to ${endpoint} failed`);
  }
  return response.blob();
}

const form = new FormData(document.querySelector('form'));
const input = await fetch('/src/img.png').then(r => r.blob());
const ref = await fetch('/src/ref.jpg').then(r => r.blob());
const lut = await fetch('/src/grading.cube').then(r => r.blob());

const c = await run('clahe', input, { clahe_clip: 2.0, tile: 8 });
const n = await run('noise', c, { noise_std: 0.02, seed: 123 });
const f = await run('fft', n, { fft_mode: 'ref', fft_variant: 'v4', seed: 123 }, { reference_image: ref });
const l = await run('lut', f, { lut_strength: 0.1 }, { lut_file: lut, output_format: 'png' });
await run('forensic-camera', l, { datetime_original: '2026-09-19 10:30:00' })
  .then(async (final) => {
    const objectUrl = URL.createObjectURL(final);
    // final is a JPEG blob; objectUrl can be used for preview or download
  });
```

## Strict Field Validation Reminders

- Every config model forbids unknown and extra fields -> HTTP `422`.
- `output_format` accepts only `png` and `jpeg`.
- `fft_mode` accepts only `auto`, `ref`, `model`.
- `lbp_method` accepts only `default`, `ror`, `uniform`, `var`.
- `fft_variant` accepts only `v1 (Original)`, `v2`, `v3`, `v4`.
- `profile` on `/forensic-camera` accepts only `iphone_16_pro`.
- `reference_image` is only meaningful on `/fft`, `/glcm`, `/lbp`, `/awb`.
- `lut_file` is only meaningful (and required) on `/lut`.
- `include_exif` defaults to `false`; ordinary endpoints turn it on explicitly
  only if the caller asks for legacy EXIF embedding.
