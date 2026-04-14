from .autowb import auto_white_balance_ref
from .clahe import clahe_color_correction
from .color_lut import load_lut, apply_lut
from .exif import remove_exif_pil
from .fourier_pipeline import fourier_match_spectrum as fourier_match_spectrum_v1
from .fourier_pipeline_new_algo import fourier_match_spectrum as fourier_match_spectrum_v2
from .fourier_pipeline_v3 import fourier_match_spectrum_v3
from .gaussian_noise import add_gaussian_noise
from .perturbation import randomized_perturbation
from .glcm_normalization import glcm_normalize
from .lbp_normalization import lbp_normalize
from .non_semantic_unmarker import attack_non_semantic
from .blend import blend_colors

# Default alias for backward compatibility
fourier_match_spectrum = fourier_match_spectrum_v2

FOURIER_VARIANTS = {
    "v1 (Original)": fourier_match_spectrum_v1,
    "v2": fourier_match_spectrum_v2,
    "v3": fourier_match_spectrum_v3,
}

__all__ = [
    'auto_white_balance_ref',
    'clahe_color_correction',
    'load_lut',
    'apply_lut',
    'remove_exif_pil',
    'fourier_match_spectrum',
    'fourier_match_spectrum_v1',
    'fourier_match_spectrum_v2',
    'fourier_match_spectrum_v3',
    'FOURIER_VARIANTS',
    'add_gaussian_noise',
    'randomized_perturbation',
    'glcm_normalize',
    'lbp_normalize',
    'attack_non_semantic',
    'blend_colors',
]