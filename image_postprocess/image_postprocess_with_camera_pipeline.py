#!/usr/bin/env python3
"""
image_postprocess_with_camera_pipeline.py

Main pipeline for image postprocessing with an optional realistic camera-pipeline simulator.
This file retains the original interface for CLI and imports, ensuring compatibility with existing code.
Imports helper functions and camera pipeline simulation from separate modules.
"""

import argparse
import os
from PIL import Image
import numpy as np
import piexif
from datetime import datetime


from .utils import add_gaussian_noise, clahe_color_correction, randomized_perturbation, fourier_match_spectrum
from .camera_pipeline import simulate_camera_pipeline

def add_fake_exif():
    """
    Generates a plausible set of fake EXIF data.
    Returns:
        bytes: The EXIF data as a byte string, ready for insertion.
    """
    # Get current time for timestamp
    now = datetime.now()
    datestamp = now.strftime("%Y:%m:%d %H:%M:%S")

    # Define some plausible fake EXIF tags
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
        piexif.ExifIFD.FocalLength: (50, 1),      # 50mm
    }
    gps_ifd = {} # Empty GPS info

    exif_dict = {"0th": zeroth_ifd, "Exif": exif_ifd, "GPS": gps_ifd, "1st": {}, "thumbnail": None}
    exif_bytes = piexif.dump(exif_dict)
    return exif_bytes

def process_image(path_in, path_out, args):
    img = Image.open(path_in).convert('RGB')
    # img = remove_exif_pil(img) # <-- This line is removed

    arr = np.array(img)

    arr = clahe_color_correction(arr, clip_limit=args.clahe_clip, tile_grid_size=(args.tile, args.tile))

    ref_arr = None
    if args.fft_ref:
        ref_img = Image.open(args.fft_ref).convert('RGB')
        ref_arr = np.array(ref_img)

    arr = fourier_match_spectrum(arr, ref_img_arr=ref_arr, mode=args.fft_mode,
                                 alpha=args.fft_alpha, cutoff=args.cutoff,
                                 strength=args.fstrength, randomness=args.randomness,
                                 phase_perturb=args.phase_perturb, radial_smooth=args.radial_smooth,
                                 seed=args.seed)

    arr = add_gaussian_noise(arr, std_frac=args.noise_std, seed=args.seed)
    arr = randomized_perturbation(arr, magnitude_frac=args.perturb, seed=args.seed)

    # call the camera simulator if requested
    if args.sim_camera:
        arr = simulate_camera_pipeline(arr,
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
                                       seed=args.seed)

    out_img = Image.fromarray(arr)
    
    # Generate fake EXIF data and save it with the image
    fake_exif_bytes = add_fake_exif()
    out_img.save(path_out, exif=fake_exif_bytes)

def build_argparser():
    p = argparse.ArgumentParser(description="Image postprocessing pipeline with camera simulation")
    p.add_argument('input', help='Input image path')
    p.add_argument('output', help='Output image path')
    p.add_argument('--ref', help='Optional reference image for color matching (not implemented)', default=None)
    p.add_argument('--noise-std', type=float, default=0.02, help='Gaussian noise std fraction of 255 (0-0.1)')
    p.add_argument('--clahe-clip', type=float, default=2.0, help='CLAHE clip limit')
    p.add_argument('--tile', type=int, default=8, help='CLAHE tile grid size')
    p.add_argument('--cutoff', type=float, default=0.25, help='Fourier cutoff (0..1)')
    p.add_argument('--fstrength', type=float, default=0.9, help='Fourier blend strength (0..1)')
    p.add_argument('--randomness', type=float, default=0.05, help='Randomness for Fourier mask modulation')
    p.add_argument('--perturb', type=float, default=0.008, help='Randomized perturb magnitude fraction (0..0.05)')
    p.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')

    # FFT-matching options
    p.add_argument('--fft-ref', help='Optional reference image for FFT spectral matching', default=None)
    p.add_argument('--fft-mode', choices=('auto','ref','model'), default='auto', help='FFT mode: auto picks ref if available')
    p.add_argument('--fft-alpha', type=float, default=1.0, help='Alpha for 1/f model (spectrum slope)')
    p.add_argument('--phase-perturb', type=float, default=0.08, help='Phase perturbation strength (radians)')
    p.add_argument('--radial-smooth', type=int, default=5, help='Radial smoothing (bins) for spectrum profiles')

    # Camera-simulator options
    p.add_argument('--sim-camera', action='store_true', help='Enable camera-pipeline simulation (Bayer, CA, vignette, JPEG cycles)')
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

    return p

if __name__ == "__main__":
    args = build_argparser().parse_args()
    if not os.path.exists(args.input):
        print("Input not found:", args.input)
        raise SystemExit(2)
    process_image(args.input, args.output, args)
    print("Saved:", args.output)