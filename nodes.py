from PIL import Image
import numpy as np
import os
import tempfile
from types import SimpleNamespace
try:
    from worker import Worker
except Exception as e:
    Worker = None
    IMPORT_ERROR = str(e)
else:
    IMPORT_ERROR = None

class NovaNodes:
    """
    ComfyUI node: Full post-processing chain using Worker from GUI
    All augmentations with tunable parameters.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),

                # EXIF
                "apply_exif_o": ("BOOLEAN", {"default": True}),

                # Noise
                "noise_std_frac": ("FLOAT", {"default": 0.015, "min": 0.0, "max": 0.1, "step": 0.001}),
                "hot_pixel_prob": ("FLOAT", {"default": 1e-6, "min": 0.0, "max": 1e-3, "step": 1e-7}),
                "perturb_mag_frac": ("FLOAT", {"default": 0.008, "min": 0.0, "max": 0.05, "step": 0.001}),

                # CLAHE
                "clahe_clip": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 10.0, "step": 0.1}),
                "clahe_grid": ("INT", {"default": 8, "min": 2, "max": 32, "step": 1}),

                # Fourier
                "apply_fourier_o": ("BOOLEAN", {"default": True}),
                "fourier_strength": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "fourier_randomness": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.5, "step": 0.01}),
                "fourier_phase_perturb": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 0.5, "step": 0.01}),
                "fourier_alpha": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 4.0, "step": 0.1}),
                "fourier_radial_smooth": ("INT", {"default": 5, "min": 0, "max": 50, "step": 1}),
                "fourier_mode": (["auto", "ref", "model"], {"default": "auto"}),

                # Vignette
                "apply_vignette_o": ("BOOLEAN", {"default": True}),
                "vignette_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),

                # Chromatic aberration
                "apply_chromatic_aberration_o": ("BOOLEAN", {"default": True}),
                "ca_shift": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),

                # Banding
                "apply_banding_o": ("BOOLEAN", {"default": True}),
                "banding_levels": ("INT", {"default": 64, "min": 2, "max": 256, "step": 1}),

                # Motion blur
                "apply_motion_blur_o": ("BOOLEAN", {"default": True}),
                "motion_blur_ksize": ("INT", {"default": 7, "min": 3, "max": 31, "step": 2}),

                # JPEG cycles
                "apply_jpeg_cycles_o": ("BOOLEAN", {"default": True}),
                "jpeg_cycles": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}),
                "jpeg_quality": ("INT", {"default": 85, "min": 10, "max": 100, "step": 1}),

                # Camera simulation
                "sim_camera": ("BOOLEAN", {"default": False}),
                "enable_bayer": ("BOOLEAN", {"default": True}),
                "iso_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 16.0, "step": 0.1}),
                "read_noise": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 50.0, "step": 0.1}),
            },
            "optional": {
                "ref_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("IMAGE", "EXIF")
    FUNCTION = "process"
    CATEGORY = "postprocessing"

    def process(self, image, ref_image=None,
                apply_exif_o=True,
                noise_std_frac=0.015,
                hot_pixel_prob=1e-6,
                perturb_mag_frac=0.008,
                clahe_clip=2.0,
                clahe_grid=8,
                apply_fourier_o=True,
                fourier_strength=0.9,
                fourier_randomness=0.05,
                fourier_phase_perturb=0.08,
                fourier_alpha=1.0,
                fourier_radial_smooth=5,
                fourier_mode="auto",
                apply_vignette_o=True,
                vignette_strength=0.5,
                apply_chromatic_aberration_o=True,
                ca_shift=1.0,
                apply_banding_o=True,
                banding_levels=64,
                apply_motion_blur_o=True,
                motion_blur_ksize=7,
                apply_jpeg_cycles_o=True,
                jpeg_cycles=2,
                jpeg_quality=85,
                sim_camera=False,
                enable_bayer=True,
                iso_scale=1.0,
                read_noise=2.0):

        if Worker is None:
            raise ImportError(f"Could not import Worker module: {IMPORT_ERROR}")

        # Ensure input image is a PIL Image
        if not isinstance(image, Image.Image):
            raise ValueError("Input image must be a PIL Image object")

        # Save input image as temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_input:
            input_path = tmp_input.name
            image.save(input_path)

        # Prepare reference image if provided
        ref_path = None
        if ref_image is not None:
            if not isinstance(ref_image, Image.Image):
                raise ValueError("Reference image must be a PIL Image object")
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_ref:
                ref_path = tmp_ref.name
                ref_image.save(ref_path)

        # Create output temporary file path
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_output:
            output_path = tmp_output.name

        # Prepare parameters for Worker
        args = SimpleNamespace(
            noise_std=noise_std_frac,
            hot_pixel_prob=hot_pixel_prob,
            perturb=perturb_mag_frac,
            clahe_clip=clahe_clip,
            tile=clahe_grid,
            fstrength=fourier_strength,
            strength=fourier_strength,
            randomness=fourier_randomness,
            phase_perturb=fourier_phase_perturb,
            alpha=fourier_alpha,
            radial_smooth=fourier_radial_smooth,
            fft_mode=fourier_mode,
            vignette_strength=vignette_strength,
            chroma_strength=ca_shift,
            banding_strength=1.0 if apply_banding_o else 0.0,
            motion_blur_kernel=motion_blur_ksize,
            jpeg_cycles=jpeg_cycles,
            jpeg_qmin=jpeg_quality,
            jpeg_qmax=jpeg_quality,
            sim_camera=sim_camera,
            no_no_bayer=enable_bayer,
            iso_scale=iso_scale,
            read_noise=read_noise,
            ref=ref_path,
            fft_ref=ref_path,
            seed=None,  # Seed handling can be added if Worker supports it
            cutoff=0.25  # Default value from GUI, adjustable if needed
        )

        # Run Worker
        worker = Worker(input_path, output_path, args)
        worker.run()  # Assuming Worker has a synchronous run() method

        # Load output image
        output_img = Image.open(output_path)

        # Handle EXIF
        new_exif = ""
        if apply_exif_o:
            output_img, new_exif = self._add_fake_exif(output_img)

        # Clean up temporary files
        os.unlink(input_path)
        if ref_path:
            os.unlink(ref_path)
        os.unlink(output_path)

        return (output_img, new_exif)

    def _add_fake_exif(self, img: Image.Image) -> tuple[Image.Image, str]:
        """Insert random but realistic camera EXIF metadata."""
        import random
        import io
        import piexif
        exif_dict = {
            "0th": {
                piexif.ImageIFD.Make: random.choice(["Canon", "Nikon", "Sony", "Fujifilm", "Olympus", "Leica"]),
                piexif.ImageIFD.Model: random.choice([
                    "EOS 5D Mark III", "D850", "Alpha 7R IV", "X-T4", "OM-D E-M1 Mark III", "Q2"
                ]),
                piexif.ImageIFD.Software: "Adobe Lightroom",
            },
            "Exif": {
                piexif.ExifIFD.FNumber: (random.randint(10, 22), 10),
                piexif.ExifIFD.ExposureTime: (1, random.randint(60, 4000)),
                piexif.ExifIFD.ISOSpeedRatings: random.choice([100, 200, 400, 800, 1600, 3200]),
                piexif.ExifIFD.FocalLength: (random.randint(24, 200), 1),
            },
        }
        exif_bytes = piexif.dump(exif_dict)
        output = io.BytesIO()
        img.save(output, format="JPEG", exif=exif_bytes)
        output.seek(0)
        return (Image.open(output), str(exif_bytes))

# -------------
#  Registration
# -------------
NODE_CLASS_MAPPINGS = {
    "NovaNodes": NovaNodes,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "NovaNodes": "Image Postprocess (NOVA NODES)",
}