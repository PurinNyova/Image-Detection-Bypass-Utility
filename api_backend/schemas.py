from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


StageName = Literal[
    'blend',
    'non_semantic',
    'clahe',
    'fft',
    'glcm',
    'lbp',
    'noise',
    'perturb',
    'sim_camera',
    'awb',
    'lut',
]

OutputFormat = Literal['png', 'jpeg']


class BlendConfig(StrictModel):
    blend_tolerance: float = 10.0
    blend_min_region: int = 50
    blend_max_samples: int = 100000
    blend_n_jobs: int | None = None


class NonSemanticConfig(StrictModel):
    ns_iterations: int = 500
    ns_learning_rate: float = 3e-4
    ns_t_lpips: float = 4e-2
    ns_t_l2: float = 3e-5
    ns_c_lpips: float = 1e-2
    ns_c_l2: float = 0.6
    ns_grad_clip: float = 0.05


class ClaheConfig(StrictModel):
    clahe_clip: float = 2.0
    tile: int = 8


class FFTConfig(StrictModel):
    fft_mode: Literal['auto', 'ref', 'model'] = 'auto'
    fft_alpha: float = 1.0
    cutoff: float = 0.25
    fstrength: float = 0.9
    randomness: float = 0.05
    phase_perturb: float = 0.08
    radial_smooth: int = 5
    fft_variant: str = 'v2'
    seed: int | None = None


class GLCMConfig(StrictModel):
    glcm_distances: list[int] = Field(default_factory=lambda: [1])
    glcm_angles: list[float] = Field(default_factory=lambda: [0.0, 0.7853981634, 1.5707963268, 2.3561944902])
    glcm_levels: int = 256
    glcm_strength: float = 0.9
    seed: int | None = None


class LBPConfig(StrictModel):
    lbp_radius: int = 3
    lbp_n_points: int = 24
    lbp_method: Literal['default', 'ror', 'uniform', 'var'] = 'uniform'
    lbp_strength: float = 0.9
    seed: int | None = None


class NoiseConfig(StrictModel):
    noise_std: float = 0.02
    seed: int | None = None


class PerturbConfig(StrictModel):
    perturb_magnitude: float = 0.008
    seed: int | None = None


class SimCameraConfig(StrictModel):
    no_no_bayer: bool = True
    jpeg_cycles: int = 1
    jpeg_qmin: int = 88
    jpeg_qmax: int = 96
    vignette_strength: float = 0.35
    chroma_strength: float = 1.2
    iso_scale: float = 1.0
    read_noise: float = 2.0
    hot_pixel_prob: float = 1e-6
    banding_strength: float = 0.0
    motion_blur_kernel: int = 1
    seed: int | None = None


class AWBConfig(StrictModel):
    seed: int | None = None


class LUTConfig(StrictModel):
    lut_strength: float = 0.1


class PipelineConfig(StrictModel):
    execution_order: bool = False
    stage_order: list[StageName] = Field(default_factory=list)
    stages: dict[StageName, dict[str, Any]] = Field(default_factory=dict)