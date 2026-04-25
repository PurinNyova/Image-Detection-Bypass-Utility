#!/usr/bin/env python3
"""
processor.py

Main pipeline for image postprocessing with an optional realistic camera-pipeline simulator.
Added support for applying 1D PNG/.npy LUTs and .cube 3D LUTs via --lut.
Added GLCM and LBP normalization using the same reference as FFT.
"""

import argparse
from io import BytesIO
import os
import sys
from PIL import Image
import numpy as np
import piexif
from datetime import datetime

from .utils import (
    add_gaussian_noise,
    clahe_color_correction,
    randomized_perturbation,
    fourier_match_spectrum,
    auto_white_balance_ref,
    load_lut,
    apply_lut,
    glcm_normalize,
    lbp_normalize,
    attack_non_semantic,
    blend_colors,
    FOURIER_VARIANTS,
)
from .camera_pipeline import simulate_camera_pipeline


DEFAULT_STAGE_ORDER = [
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

ARG_DEFAULTS = {
    'input': None,
    'output': None,
    'awb': False,
    'ref': None,
    'noise_std': 0.02,
    'clahe_clip': 2.0,
    'tile': 8,
    'cutoff': 0.25,
    'fstrength': 0.9,
    'randomness': 0.05,
    'seed': None,
    'fft_ref': None,
    'fft_mode': 'auto',
    'fft_alpha': 1.0,
    'phase_perturb': 0.08,
    'radial_smooth': 5,
    'fft_variant': 'v2',
    'glcm': False,
    'glcm_distances': [1],
    'glcm_angles': [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
    'glcm_levels': 256,
    'glcm_strength': 0.9,
    'lbp': False,
    'lbp_radius': 3,
    'lbp_n_points': 24,
    'lbp_method': 'uniform',
    'lbp_strength': 0.9,
    'non_semantic': False,
    'ns_iterations': 500,
    'ns_learning_rate': 3e-4,
    'ns_t_lpips': 4e-2,
    'ns_t_l2': 3e-5,
    'ns_c_lpips': 1e-2,
    'ns_c_l2': 0.6,
    'ns_grad_clip': 0.05,
    'sim_camera': False,
    'no_no_bayer': True,
    'jpeg_cycles': 1,
    'jpeg_qmin': 88,
    'jpeg_qmax': 96,
    'vignette_strength': 0.35,
    'chroma_strength': 1.2,
    'iso_scale': 1.0,
    'read_noise': 2.0,
    'hot_pixel_prob': 1e-6,
    'banding_strength': 0.0,
    'motion_blur_kernel': 1,
    'lut': None,
    'lut_strength': 0.1,
    'execution_order': False,
    'noise': False,
    'clahe': False,
    'fft': False,
    'perturb': False,
    'perturb_magnitude': 0.008,
    'blend': False,
    'blend_tolerance': 10.0,
    'blend_min_region': 50,
    'blend_max_samples': 100000,
    'blend_n_jobs': None,
    'include_exif': True,
    '_raw_argv': None,
    '_stage_order': None,
    '_lut_data': None,
}

MASTER_STAGE_FLAGS = {
    '--blend': 'blend',
    '--non-semantic': 'non_semantic',
    '--clahe': 'clahe',
    '--fft': 'fft',
    '--glcm': 'glcm',
    '--lbp': 'lbp',
    '--noise': 'noise',
    '--perturb': 'perturb',
    '--sim-camera': 'sim_camera',
    '--awb': 'awb',
    '--lut': 'lut',
}


def is_stage_enabled(args, stage_name):
    if stage_name == 'lut':
        return bool(getattr(args, 'lut', None))
    return bool(getattr(args, stage_name, False))


def build_processing_args(
    enabled_stages=None,
    stage_config=None,
    execution_order=False,
    raw_argv=None,
    stage_order=None,
    input_path=None,
    output_path=None,
    ref_path=None,
    fft_ref_path=None,
    lut_path=None,
    lut_data=None,
    include_exif=True,
):
    values = dict(ARG_DEFAULTS)
    values['input'] = input_path
    values['output'] = output_path
    values['ref'] = ref_path
    values['fft_ref'] = fft_ref_path
    values['include_exif'] = include_exif
    values['execution_order'] = execution_order
    values['_raw_argv'] = list(raw_argv) if raw_argv is not None else None
    values['_stage_order'] = list(stage_order) if stage_order is not None else None
    values['_lut_data'] = lut_data

    if stage_config:
        values.update(stage_config)

    enabled_stage_set = set(enabled_stages or [])
    for stage_name in DEFAULT_STAGE_ORDER:
        if stage_name == 'lut':
            continue
        if stage_name in enabled_stage_set:
            values[stage_name] = True

    if lut_path is not None:
        values['lut'] = lut_path

    return argparse.Namespace(**values)


def resolve_stage_order(args):
    if not getattr(args, 'execution_order', False):
        return list(DEFAULT_STAGE_ORDER)

    explicit_stage_order = getattr(args, '_stage_order', None)
    if explicit_stage_order is not None:
        return list(explicit_stage_order)

    raw_argv = getattr(args, '_raw_argv', None) or []
    ordered_stages = []
    seen = set()

    for token in raw_argv:
        if not token.startswith('--'):
            continue
        flag = token.split('=', 1)[0]
        stage_name = MASTER_STAGE_FLAGS.get(flag)
        if stage_name and stage_name not in seen:
            ordered_stages.append(stage_name)
            seen.add(stage_name)

    for stage_name in DEFAULT_STAGE_ORDER:
        if stage_name not in seen:
            ordered_stages.append(stage_name)

    return ordered_stages


def add_fake_exif():
    """
    Generates a plausible set of fake EXIF data.
    Returns:
        bytes: The EXIF data as a byte string, ready for insertion.
    """
    now = datetime.now()
    datestamp = now.strftime("%Y:%m:%d %H:%M:%S")

    zeroth_ifd = {
        piexif.ImageIFD.Make: b"PurinCamera",
        piexif.ImageIFD.Model: b"Model420X",
        piexif.ImageIFD.Software: b"NovaImageProcessor",
        piexif.ImageIFD.DateTime: datestamp.encode('utf-8'),
    }
    exif_ifd = {
        piexif.ExifIFD.DateTimeOriginal: datestamp.encode('utf-8'),
        piexif.ExifIFD.DateTimeDigitized: datestamp.encode('utf-8'),
        piexif.ExifIFD.ExposureTime: (1, 125),  # 1/125s
        piexif.ExifIFD.FNumber: (28, 10),      # F/2.8
        piexif.ExifIFD.ISOSpeedRatings: 200,
        piexif.ExifIFD.FocalLength: (50, 1),    # 50mm
    }
    gps_ifd = {}

    exif_dict = {"0th": zeroth_ifd, "Exif": exif_ifd, "GPS": gps_ifd, "1st": {}, "thumbnail": None}
    exif_bytes = piexif.dump(exif_dict)
    return exif_bytes


def encode_image_array(arr, output_format='PNG', include_exif=True):
    normalized_format = output_format.upper()
    if normalized_format == 'JPG':
        normalized_format = 'JPEG'
    if normalized_format not in ('PNG', 'JPEG'):
        raise ValueError(f"Unsupported output format: {output_format}")

    out_img = Image.fromarray(arr)
    fake_exif_bytes = add_fake_exif() if include_exif else None
    save_kwargs = {'format': normalized_format}
    if fake_exif_bytes is not None:
        save_kwargs['exif'] = fake_exif_bytes

    buffer = BytesIO()
    out_img.save(buffer, **save_kwargs)
    media_type = 'image/jpeg' if normalized_format == 'JPEG' else 'image/png'
    return buffer.getvalue(), media_type


def save_image_array(arr, path_out, include_exif=True):
    out_img = Image.fromarray(arr)
    if include_exif:
        out_img.save(path_out, exif=add_fake_exif())
    else:
        out_img.save(path_out)


def load_image_array(path):
    return np.array(Image.open(path).convert('RGB'))


def process_array(arr, args, ref_arr_awb=None, ref_arr_fft=None):
    arr = np.array(arr, copy=True)

    if ref_arr_fft is None and getattr(args, 'fft_ref', None):
        try:
            ref_arr_fft = load_image_array(args.fft_ref)
        except Exception as e:
            print(f"Warning: failed to load FFT reference '{args.fft_ref}': {e}. Skipping FFT reference matching.")
            ref_arr_fft = None

    if ref_arr_awb is None and getattr(args, 'ref', None):
        try:
            ref_arr_awb = load_image_array(args.ref)
        except Exception as e:
            print(f"Warning: failed to load AWB reference '{args.ref}': {e}. Skipping AWB.")
            ref_arr_awb = None

    def run_blend(current_arr):
        if not args.blend:
            return current_arr
        try:
            return blend_colors(
                current_arr,
                tolerance=args.blend_tolerance,
                min_region_size=args.blend_min_region,
                max_kmeans_samples=args.blend_max_samples,
                n_jobs=args.blend_n_jobs,
            )
        except Exception as e:
            print(f"Warning: Blending failed: {e}. Skipping blending.")
            return current_arr

    def run_non_semantic(current_arr):
        if not args.non_semantic:
            return current_arr
        print("Applying non-semantic attack...")
        try:
            return attack_non_semantic(
                current_arr,
                iterations=args.ns_iterations,
                learning_rate=args.ns_learning_rate,
                t_lpips=args.ns_t_lpips,
                t_l2=args.ns_t_l2,
                c_lpips=args.ns_c_lpips,
                c_l2=args.ns_c_l2,
                grad_clip_value=args.ns_grad_clip,
            )
        except Exception as e:
            print(f"Warning: Non-semantic attack failed: {e}. Skipping non-semantic attack.")
            return current_arr

    def run_clahe(current_arr):
        if not args.clahe:
            return current_arr
        return clahe_color_correction(current_arr, clip_limit=args.clahe_clip, tile_grid_size=(args.tile, args.tile))

    def run_fft(current_arr):
        if not args.fft:
            return current_arr
        fft_variant = getattr(args, 'fft_variant', 'v2')
        fft_func = FOURIER_VARIANTS.get(fft_variant, fourier_match_spectrum)
        fft_kwargs = dict(
            ref_img_arr=ref_arr_fft,
            mode=args.fft_mode,
            alpha=args.fft_alpha,
            cutoff=args.cutoff,
            strength=args.fstrength,
            randomness=args.randomness,
            seed=args.seed,
            radial_smooth=args.radial_smooth,
        )
        if fft_variant != 'v3':
            fft_kwargs['phase_perturb'] = args.phase_perturb
        return fft_func(current_arr, **fft_kwargs)

    def run_glcm(current_arr):
        if not args.glcm:
            return current_arr
        return glcm_normalize(
            current_arr,
            ref_img_arr=ref_arr_fft,
            distances=args.glcm_distances,
            angles=args.glcm_angles,
            levels=args.glcm_levels,
            strength=args.glcm_strength,
            seed=args.seed,
        )

    def run_lbp(current_arr):
        if not args.lbp:
            return current_arr
        return lbp_normalize(
            current_arr,
            ref_img_arr=ref_arr_fft,
            radius=args.lbp_radius,
            n_points=args.lbp_n_points,
            method=args.lbp_method,
            strength=args.lbp_strength,
            seed=args.seed,
        )

    def run_noise(current_arr):
        if not args.noise:
            return current_arr
        return add_gaussian_noise(current_arr, std_frac=args.noise_std, seed=args.seed)

    def run_perturb(current_arr):
        if not args.perturb:
            return current_arr
        return randomized_perturbation(current_arr, magnitude_frac=args.perturb_magnitude, seed=args.seed)

    def run_sim_camera(current_arr):
        if not args.sim_camera:
            return current_arr
        return simulate_camera_pipeline(
            current_arr,
            bayer=not args.no_no_bayer,
            jpeg_cycles=args.jpeg_cycles,
            jpeg_quality_range=(args.jpeg_qmin, args.jpeg_qmax),
            vignette_strength=args.vignette_strength,
            chroma_aberr_strength=args.chroma_strength,
            iso_scale=args.iso_scale,
            read_noise_std=args.read_noise,
            hot_pixel_prob=args.hot_pixel_prob,
            banding_strength=args.banding_strength,
            motion_blur_kernel=args.motion_blur_kernel,
            seed=args.seed,
        )

    def run_awb(current_arr):
        if not args.awb:
            return current_arr
        if ref_arr_awb is not None:
            return auto_white_balance_ref(current_arr, ref_arr_awb)
        print("Applying AWB using grey-world assumption...")
        return auto_white_balance_ref(current_arr, None)

    def run_lut(current_arr):
        if not args.lut:
            return current_arr
        try:
            lut = getattr(args, '_lut_data', None)
            if lut is None:
                lut = load_lut(args.lut)
            arr_uint8 = np.clip(current_arr, 0, 255).astype(np.uint8)
            arr_lut = apply_lut(arr_uint8, lut, strength=args.lut_strength)
            return np.clip(arr_lut, 0, 255).astype(np.uint8)
        except Exception as e:
            print(f"Warning: failed to load/apply LUT '{args.lut}': {e}. Skipping LUT.")
            return current_arr

    stage_handlers = {
        'blend': run_blend,
        'non_semantic': run_non_semantic,
        'clahe': run_clahe,
        'fft': run_fft,
        'glcm': run_glcm,
        'lbp': run_lbp,
        'noise': run_noise,
        'perturb': run_perturb,
        'sim_camera': run_sim_camera,
        'awb': run_awb,
        'lut': run_lut,
    }

    for stage_name in resolve_stage_order(args):
        arr = stage_handlers[stage_name](arr)

    return arr


def process_image(path_in, path_out, args):
    arr = load_image_array(path_in)
    arr = process_array(arr, args)
    save_image_array(arr, path_out, include_exif=getattr(args, 'include_exif', True))


def build_argparser():
    p = argparse.ArgumentParser(description="Image postprocessing pipeline with camera simulation, LUT support, GLCM, and LBP normalization")
    p.add_argument('input', help='Input image path')
    p.add_argument('output', help='Output image path')

    # AWB Options
    p.add_argument('--awb', action='store_true', default=False, help='Enable automatic white balancing. Uses grey-world if --ref is not provided.')
    p.add_argument('--ref', help='Optional reference image for auto white-balance (only used if --awb is enabled)', default=None)
    
    p.add_argument('--noise-std', type=float, default=0.02, help='Gaussian noise std fraction of 255 (0-0.1)')
    p.add_argument('--clahe-clip', type=float, default=2.0, help='CLAHE clip limit')
    p.add_argument('--tile', type=int, default=8, help='CLAHE tile grid size')
    p.add_argument('--cutoff', type=float, default=0.25, help='Fourier cutoff (0..1)')
    p.add_argument('--fstrength', type=float, default=0.9, help='Fourier blend strength (0..1)')
    p.add_argument('--randomness', type=float, default=0.05, help='Randomness for Fourier mask modulation')
    p.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')

    # FFT-matching options
    p.add_argument('--fft-ref', help='Optional reference image for FFT spectral matching, GLCM, and LBP', default=None)
    p.add_argument('--fft-mode', choices=('auto','ref','model'), default='auto', help='FFT mode: auto picks ref if available')
    p.add_argument('--fft-alpha', type=float, default=1.0, help='Alpha for 1/f model (spectrum slope)')
    p.add_argument('--phase-perturb', type=float, default=0.08, help='Phase perturbation strength (radians)')
    p.add_argument('--radial-smooth', type=int, default=5, help='Radial smoothing (bins) for spectrum profiles')

    # GLCM normalization options
    p.add_argument('--glcm', action='store_true', default=False, help='Enable GLCM normalization using FFT reference if available')
    p.add_argument('--glcm-distances', type=int, nargs='+', default=[1], help='Distances for GLCM computation')
    p.add_argument('--glcm-angles', type=float, nargs='+', default=[0, np.pi/4, np.pi/2, 3*np.pi/4], help='Angles for GLCM computation (in radians)')
    p.add_argument('--glcm-levels', type=int, default=256, help='Number of gray levels for GLCM')
    p.add_argument('--glcm-strength', type=float, default=0.9, help='Strength of GLCM feature matching (0..1)')

    # LBP normalization options
    p.add_argument('--lbp', action='store_true', default=False, help='Enable LBP normalization using FFT reference if available')
    p.add_argument('--lbp-radius', type=int, default=3, help='Radius of LBP operator')
    p.add_argument('--lbp-n-points', type=int, default=24, help='Number of circularly symmetric neighbor set points for LBP')
    p.add_argument('--lbp-method', choices=('default', 'ror', 'uniform', 'var'), default='uniform', help='LBP method')
    p.add_argument('--lbp-strength', type=float, default=0.9, help='Strength of LBP histogram matching (0..1)')

    # Non-semantic attack options
    p.add_argument('--non-semantic', action='store_true', default=False, help='Apply non-semantic attack on the image')
    p.add_argument('--ns-iterations', type=int, default=500, help='Iterations for non-semantic attack')
    p.add_argument('--ns-learning-rate', type=float, default=3e-4, help='Learning rate for non-semantic attack')
    p.add_argument('--ns-t-lpips', type=float, default=4e-2, help='LPIPS threshold for non-semantic attack')
    p.add_argument('--ns-t-l2', type=float, default=3e-5, help='L2 threshold for non-semantic attack')
    p.add_argument('--ns-c-lpips', type=float, default=1e-2, help='LPIPS constant for non-semantic attack')
    p.add_argument('--ns-c-l2', type=float, default=0.6, help='L2 constant for non-semantic attack')
    p.add_argument('--ns-grad-clip', type=float, default=0.05, help='Gradient clipping value for non-semantic attack')

    # Camera-simulator options
    p.add_argument('--sim-camera', action='store_true', default=False, help='Enable camera-pipeline simulation (Bayer, CA, vignette, JPEG cycles)')
    p.add_argument('--no-no-bayer', dest='no_no_bayer', action='store_false', help='Disable Bayer/demosaic step (double negative kept for backward compat)')
    p.set_defaults(no_no_bayer=True)
    p.add_argument('--jpeg-cycles', type=int, default=1, help='Number of JPEG recompression cycles to apply')
    p.add_argument('--jpeg-qmin', type=int, default=88, help='Min JPEG quality for recompression')
    p.add_argument('--jpeg-qmax', type=int, default=96, help='Max JPEG quality for recompression')
    p.add_argument('--vignette-strength', type=float, default=0.35, help='Vignette strength (0..1)')
    p.add_argument('--chroma-strength', type=float, default=1.2, help='Chromatic aberration strength (pixels)')
    p.add_argument('--iso-scale', type=float, default=1.0, help='ISO/exposure scale for Poisson noise')
    p.add_argument('--read-noise', type=float, default=2.0, help='Read noise sigma for sensor noise')
    p.add_argument('--hot-pixel-prob', type=float, default=1e-6, help='Per-pixel probability of hot pixel')
    p.add_argument('--banding-strength', type=float, default=0.0, help='Horizontal banding amplitude (0..1)')
    p.add_argument('--motion-blur-kernel', type=int, default=1, help='Motion blur kernel size (1 = none)')

    # LUT options
    p.add_argument('--lut', type=str, default=None, help='Path to a 1D PNG (256x1) or .npy LUT, or a .cube 3D LUT')
    p.add_argument('--lut-strength', type=float, default=0.1, help='Strength to blend LUT (0.0 = no effect, 1.0 = full LUT)')
    p.add_argument('--execution-order', action='store_true', default=False,
                   help='Execute enabled master stages in the order their flags appear in the CLI call')

    # New positive flags to enable utils functions
    p.add_argument('--noise', action='store_true', default=False, help='Enable Gaussian noise addition')
    p.add_argument('--clahe', action='store_true', default=False, help='Enable CLAHE color correction')
    p.add_argument('--fft', action='store_true', default=False, help='Enable FFT spectral matching')
    p.add_argument('--perturb', action='store_true', default=False, help='Enable randomized perturbation')
    p.add_argument('--perturb-magnitude', type=float, default=0.008, help='Randomized perturb magnitude fraction (0..0.05)')

    # Blending options
    p.add_argument('--blend', action='store_true', default=False, help='Enable color')
    p.add_argument('--blend-tolerance', type=float, default=10.0, help='Color tolerance for blending (smaller = more colors)')
    p.add_argument('--blend-min-region', type=int, default=50, help='Minimum region size to retain (in pixels)')
    p.add_argument('--blend-max-samples', type=int, default=100000, help='Maximum pixels to sample for k-means (for speed)')
    p.add_argument('--blend-n-jobs', type=int, default=None, help='Number of worker threads for blending (default: os.cpu_count())')

    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    args._raw_argv = sys.argv[1:]
    if not os.path.exists(args.input):
        print("Input not found:", args.input)
        raise SystemExit(2)
    process_image(args.input, args.output, args)
    print("Saved:", args.output)