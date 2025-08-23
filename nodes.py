import torch
from PIL import Image
import numpy as np
import os
import tempfile
from types import SimpleNamespace
from typing import Tuple
try:
    from .image_postprocess import process_image
except Exception as e:
    process_image = None
    IMPORT_ERROR = str(e)
else:
    IMPORT_ERROR = None


class NovaNodes:
    """
    ComfyUI node: Full post-processing chain using process_image from image_postprocess
    All augmentations with tunable parameters.

    NOTE: Adjusted to match FOOLAI output:
      - Returns an IMAGE as a single PyTorch tensor shaped (1, H, W, C), dtype=float32, values in [0.0, 1.0].
      - Returns EXIF as a STRING (second output slot).
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

        if process_image is None:
            raise ImportError(f"Could not import process_image function: {IMPORT_ERROR}")

        tmp_files = []

        def to_pil_from_any(inp):
            """Convert a torch tensor / numpy array of many shapes into a PIL RGB Image."""
            # get numpy
            if isinstance(inp, torch.Tensor):
                arr = inp.detach().cpu().numpy()
            else:
                arr = np.asarray(inp)

            # remove leading batch dimension if present
            if arr.ndim == 4 and arr.shape[0] == 1:
                arr = arr[0]

            # CHW -> HWC
            if arr.ndim == 3 and arr.shape[0] in (1, 3):
                arr = np.transpose(arr, (1, 2, 0))

            # if still 3D and last dim is channel (H,W,C) but C==1 or 3: OK
            if arr.ndim == 2:
                # grayscale HxW -> make HxWx1
                arr = arr[:, :, None]

            # Now arr should be H x W x C
            if arr.ndim != 3:
                # try permutations heuristically (rare)
                for perm in [(1, 2, 0), (2, 0, 1), (0, 2, 1)]:
                    try:
                        cand = np.transpose(arr, perm)
                        if cand.ndim == 3:
                            arr = cand
                            break
                    except Exception:
                        pass

            if arr.ndim != 3:
                raise TypeError(f"Cannot convert array to HWC image, final ndim={arr.ndim}, shape={arr.shape}")

            # Normalize numeric range to 0..255 uint8
            if np.issubdtype(arr.dtype, np.floating):
                # assume floats are 0..1 if max <= 1.0
                if arr.max() <= 1.0:
                    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)

            # If single channel, replicate to 3 channels (we want RGB files)
            if arr.shape[2] == 1:
                arr = np.repeat(arr, 3, axis=2)

            # finally create PIL
            return Image.fromarray(arr)

        try:
            # ---- Input image -> temporary input file ----
            pil_img = to_pil_from_any(image[0])
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_input:
                input_path = tmp_input.name
                pil_img.save(input_path)
                tmp_files.append(input_path)

            # ---- Reference image for AWB and FFT if present ----
            ref_path = None
            if ref_image is not None:
                pil_ref = to_pil_from_any(ref_image[0])
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_ref:
                    ref_path = tmp_ref.name
                    pil_ref.save(ref_path)
                    tmp_files.append(ref_path)

            # ---- Output path ----
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_output:
                output_path = tmp_output.name
                tmp_files.append(output_path)

            # Prepare args for process_image (keeping your names)
            args = SimpleNamespace(
                input=input_path,
                output=output_path,
                ref=ref_path,  # Used for AWB if provided
                noise_std=noise_std_frac,
                hot_pixel_prob=hot_pixel_prob,
                perturb=perturb_mag_frac,
                clahe_clip=clahe_clip,
                tile=clahe_grid,
                fstrength=fourier_strength if apply_fourier_o else 0.0,
                randomness=fourier_randomness,
                phase_perturb=fourier_phase_perturb,
                fft_alpha=fourier_alpha,
                radial_smooth=fourier_radial_smooth,
                fft_mode=fourier_mode,
                fft_ref=ref_path,  # Used for FFT if provided
                vignette_strength=vignette_strength if apply_vignette_o else 0.0,
                chroma_strength=ca_shift if apply_chromatic_aberration_o else 0.0,
                banding_strength=1.0 if apply_banding_o else 0.0,
                motion_blur_kernel=motion_blur_ksize if apply_motion_blur_o else 1,
                jpeg_cycles=jpeg_cycles if apply_jpeg_cycles_o else 1,
                jpeg_qmin=jpeg_quality,
                jpeg_qmax=jpeg_quality,
                sim_camera=sim_camera,
                no_no_bayer=enable_bayer,
                iso_scale=iso_scale,
                read_noise=read_noise,
                seed=None,
                cutoff=0.25
            )

            # ---- Run the processing function ----
            process_image(input_path, output_path, args)

            # ---- Load result (force RGB to avoid unexpected single-channel shapes) ----
            output_img = Image.open(output_path).convert("RGB")
            img_out = np.array(output_img)  # H x W x 3, uint8

            # ---- EXIF insertion (optional) ----
            new_exif = ""
            if apply_exif_o:
                try:
                    output_img_with_exif, new_exif = self._add_fake_exif(output_img)
                    output_img = output_img_with_exif
                    img_out = np.array(output_img.convert("RGB"))
                except Exception:
                    new_exif = ""

            # ---- Convert to FOOLAI-style tensor: (1, H, W, C), float32 in [0,1] ----
            img_float = img_out.astype(np.float32) / 255.0  # H x W x C
            tensor_out = torch.from_numpy(img_float).to(dtype=torch.float32).unsqueeze(0)  # 1 x H x W x C
            tensor_out = torch.clamp(tensor_out, 0.0, 1.0)

            # Return the same format FOOLAI uses: (tensor, exif_string)
            return (tensor_out, new_exif)

        finally:
            for p in tmp_files:
                try:
                    os.unlink(p)
                except Exception:
                    pass

    def _add_fake_exif(self, img: Image.Image) -> Tuple[Image.Image, str]:
        """Insert random but realistic camera EXIF metadata."""
        import random
        import io
        try:
            import piexif
        except Exception:
            raise

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