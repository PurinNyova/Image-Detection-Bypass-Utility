export type EndpointId =
  | "color-blend"
  | "non-semantic"
  | "fft"
  | "glcm"
  | "noise"
  | "perturb"
  | "sim-camera"
  | "awb"
  | "lut"
  | "forensic-camera";

export type FieldKind =
  | "number"
  | "integer"
  | "boolean"
  | "text"
  | "select"
  | "int-array"
  | "number-array";

export type SelectOption = { value: string; label: string };

export type FieldDef = {
  key: string;
  kind: FieldKind;
  default: number | string | boolean | number[] | null;
  nullable?: boolean;
  options?: SelectOption[];
  label?: string;
  note?: string;
};

export type EndpointDef = {
  id: EndpointId;
  label: string;
  finalizer: boolean;
  referenceUpload?: boolean;
  lutUpload?: boolean;
  fields: FieldDef[];
};

const SEED: FieldDef = {
  key: "seed",
  kind: "integer",
  default: null,
  nullable: true,
  label: "seed",
  note: "Leave empty to omit (null).",
};

export const ENDPOINTS: EndpointDef[] = [
  {
    id: "color-blend",
    label: "Color Blend",
    finalizer: false,
    fields: [
      { key: "blend_tolerance", kind: "number", default: 10.0, label: "blend_tolerance" },
      { key: "blend_min_region", kind: "integer", default: 50, label: "blend_min_region" },
      { key: "blend_max_samples", kind: "integer", default: 100000, label: "blend_max_samples" },
      {
        key: "blend_n_jobs",
        kind: "integer",
        default: null,
        nullable: true,
        label: "blend_n_jobs",
        note: "Empty = null (default).",
      },
    ],
  },
  {
    id: "non-semantic",
    label: "AI Normalizer",
    finalizer: false,
    fields: [
      { key: "ns_iterations", kind: "integer", default: 500, label: "ns_iterations" },
      { key: "ns_learning_rate", kind: "number", default: 0.0003, label: "ns_learning_rate" },
      { key: "ns_t_lpips", kind: "number", default: 0.04, label: "ns_t_lpips" },
      { key: "ns_t_l2", kind: "number", default: 0.00003, label: "ns_t_l2" },
      { key: "ns_c_lpips", kind: "number", default: 0.01, label: "ns_c_lpips" },
      { key: "ns_c_l2", kind: "number", default: 0.6, label: "ns_c_l2" },
      { key: "ns_grad_clip", kind: "number", default: 0.05, label: "ns_grad_clip" },
      { key: "ns_adaptive_c_lpips", kind: "boolean", default: true, label: "ns_adaptive_c_lpips" },
      { key: "ns_c_lpips_min", kind: "number", default: 0.0001, label: "ns_c_lpips_min" },
      { key: "ns_c_lpips_max", kind: "number", default: 1.0, label: "ns_c_lpips_max" },
      { key: "ns_search_interval", kind: "integer", default: 40, label: "ns_search_interval" },
    ],
  },
  {
    id: "fft",
    label: "FFT Matching",
    finalizer: false,
    referenceUpload: true,
    fields: [
      {
        key: "fft_mode",
        kind: "select",
        default: "auto",
        label: "fft_mode",
        options: [
          { value: "auto", label: "auto" },
          { value: "ref", label: "ref" },
          { value: "model", label: "model" },
        ],
        note: "ref mode is meaningful only with a reference image uploaded.",
      },
      { key: "fft_alpha", kind: "number", default: 1.0, label: "fft_alpha" },
      { key: "cutoff", kind: "number", default: 0.25, label: "cutoff" },
      { key: "fstrength", kind: "number", default: 0.9, label: "fstrength" },
      { key: "randomness", kind: "number", default: 0.05, label: "randomness" },
      {
        key: "phase_perturb",
        kind: "number",
        default: 0.08,
        label: "phase_perturb",
        note: "Ignored by variants v3 and v4.",
      },
      { key: "radial_smooth", kind: "integer", default: 5, label: "radial_smooth" },
      {
        key: "fft_variant",
        kind: "select",
        default: "v2",
        label: "fft_variant",
        options: [
          { value: "v1 (Original)", label: "v1 (Original)" },
          { value: "v2", label: "v2" },
          { value: "v3", label: "v3" },
          { value: "v4", label: "v4" },
        ],
      },
      SEED,
    ],
  },
  {
    id: "glcm",
    label: "GLCM",
    finalizer: false,
    referenceUpload: true,
    fields: [
      {
        key: "glcm_distances",
        kind: "int-array",
        default: [1],
        label: "glcm_distances",
        note: "Comma-separated integers, e.g. 1, 2.",
      },
      {
        key: "glcm_angles",
        kind: "number-array",
        default: [0.0, 0.7853981634, 1.5707963268, 2.3561944902],
        label: "glcm_angles",
        note: "Comma-separated numbers (angles in radians).",
      },
      { key: "glcm_levels", kind: "integer", default: 256, label: "glcm_levels" },
      { key: "glcm_strength", kind: "number", default: 0.9, label: "glcm_strength" },
      SEED,
    ],
  },
  {
    id: "noise",
    label: "Noise",
    finalizer: false,
    fields: [
      { key: "noise_std", kind: "number", default: 0.02, label: "noise_std" },
      SEED,
    ],
  },
  {
    id: "perturb",
    label: "Perturb",
    finalizer: false,
    fields: [
      { key: "perturb_magnitude", kind: "number", default: 0.008, label: "perturb_magnitude" },
      SEED,
    ],
  },
  {
    id: "sim-camera",
    label: "Simulate Camera",
    finalizer: false,
    fields: [
      {
        key: "no_no_bayer",
        kind: "boolean",
        default: true,
        label: "Disable Bayer simulation (no_no_bayer)",
      },
      { key: "jpeg_cycles", kind: "integer", default: 1, label: "jpeg_cycles" },
      { key: "jpeg_qmin", kind: "integer", default: 88, label: "jpeg_qmin" },
      { key: "jpeg_qmax", kind: "integer", default: 96, label: "jpeg_qmax" },
      { key: "vignette_strength", kind: "number", default: 0.35, label: "vignette_strength" },
      { key: "chroma_strength", kind: "number", default: 1.2, label: "chroma_strength" },
      { key: "iso_scale", kind: "number", default: 1.0, label: "iso_scale" },
      { key: "read_noise", kind: "number", default: 2.0, label: "read_noise" },
      { key: "hot_pixel_prob", kind: "number", default: 0.000001, label: "hot_pixel_prob" },
      { key: "banding_strength", kind: "number", default: 0.0, label: "banding_strength" },
      { key: "motion_blur_kernel", kind: "integer", default: 1, label: "motion_blur_kernel" },
      SEED,
    ],
  },
  {
    id: "awb",
    label: "Auto White Balance",
    finalizer: false,
    referenceUpload: true,
    fields: [
      SEED,
    ],
  },
  {
    id: "lut",
    label: "LUT",
    finalizer: false,
    lutUpload: true,
    fields: [{ key: "lut_strength", kind: "number", default: 0.1, label: "lut_strength" }],
  },
  {
    id: "forensic-camera",
    label: "Forensic Camera",
    finalizer: true,
    fields: [
      {
        key: "profile",
        kind: "select",
        default: "iphone_16_pro",
        label: "profile",
        options: [{ value: "iphone_16_pro", label: "iphone_16_pro" }],
      },
      { key: "software", kind: "text", default: null, nullable: true, label: "software" },
      {
        key: "datetime_original",
        kind: "text",
        default: null,
        nullable: true,
        label: "datetime_original",
        note: "Format: YYYY-MM-DD HH:MM:SS or YYYY:MM:DD HH:MM:SS; other formats are rejected.",
      },
      { key: "iso", kind: "integer", default: null, nullable: true, label: "iso" },
      { key: "gps_lat", kind: "number", default: null, nullable: true, label: "gps_lat" },
      { key: "gps_lon", kind: "number", default: null, nullable: true, label: "gps_lon" },
      { key: "gps_alt", kind: "number", default: null, nullable: true, label: "gps_alt" },
      {
        key: "ela_flatten",
        kind: "boolean",
        default: true,
        label: "ela_flatten",
        note: "Whole-frame flatten so error-level analysis looks like a camera JPEG.",
      },
      {
        key: "strip_fingerprints",
        kind: "boolean",
        default: true,
        label: "strip_fingerprints",
        note: "Strips JFIF/XMP and other telltale metadata.",
      },
      SEED,
    ],
  },
];

export function getEndpoint(id: EndpointId): EndpointDef | undefined {
  return ENDPOINTS.find((e) => e.id === id);
}
