import torch
from PIL import Image
import numpy as np
import os
import tempfile
from types import SimpleNamespace
from typing import Tuple
import json

try:
    from .image_postprocess import process_image
except Exception as e:
    process_image = None
    IMPORT_ERROR = str(e)
else:
    IMPORT_ERROR = None

from .nodes_utils import CameraOptionsNode, NSOptionsNode

lut_extensions = ['png','npy','cube']

# ---------- Helper utilities (kept from original) ----------

def to_pil_from_any(inp):
    """Convert a torch tensor / numpy array of many shapes into a PIL RGB Image."""
    if isinstance(inp, torch.Tensor):
        arr = inp.detach().cpu().numpy()
    else:
        arr = np.asarray(inp)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise TypeError(f"Cannot convert array to HWC image, final ndim={arr.ndim}, shape={arr.shape}")
    if np.issubdtype(arr.dtype, np.floating):
        if arr.max() <= 1.0:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    return Image.fromarray(arr)

# utility parsers for list-like UI inputs

def _parse_int_list(val):
    if isinstance(val, (list, tuple)):
        return [int(x) for x in val]
    if isinstance(val, (int, np.integer)):
        return [int(val)]
    s = str(val).strip()
    if s == "":
        return []
    parts = [p for p in s.replace(',', ' ').split() if p != ""]
    return [int(p) for p in parts]


def _parse_float_list(val):
    if isinstance(val, (list, tuple)):
        return [float(x) for x in val]
    if isinstance(val, (float, int, np.floating, np.integer)):
        return [float(val)]
    s = str(val).strip()
    if s == "":
        return []
    parts = [p for p in s.replace(',', ' ').split() if p != ""]
    return [float(p) for p in parts]

class NovaNodes:
    """
    ComfyUI node: Full post-processing chain using process_image from image_postprocess.
    This version expects two optional JSON inputs:
      - Cam_Opt: JSON string produced by CameraOptionsNode
      - NS_Opt: JSON string produced by NSOptionsNode

    If those are empty, default values will be used (matching prior defaults).
    """

    @classmethod
    def INPUT_TYPES(s):
        # Keep most of the core image-processing parameters here; camera/NS options have been moved out.
        return {
            "required": {
                "image": ("IMAGE",),

                # High-level toggles for using the external nodes
                "Cam_Opt": ("CAMERAOPT", ),
                "NS_Opt": ("NONSEMANTICOP", ),

                # Parameters (noise / clahe / fourier / etc.)
                "apply_noise_o": ("BOOLEAN", {"default": True}),
                "noise_std_frac": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.1, "step": 0.001}),
                "apply_clahe_o": ("BOOLEAN", {"default": True}),
                "clahe_clip": ("FLOAT", {"default": 2.00, "min": 0.5, "max": 10.0, "step": 0.1}),
                "clahe_grid": ("INT", {"default": 8, "min": 2, "max": 32, "step": 1}),
                "fourier_cutoff": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "apply_fourier_o": ("BOOLEAN", {"default": True}),
                "fourier_strength": ("FLOAT", {"default": 0.90, "min": 0.0, "max": 1.0, "step": 0.01}),
                "fourier_randomness": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.5, "step": 0.01}),
                "fourier_phase_perturb": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 0.5, "step": 0.01}),
                "fourier_radial_smooth": ("INT", {"default": 5, "min": 0, "max": 50, "step": 1}),
                "fourier_mode": (["auto", "ref", "model"], {"default": "auto"}),
                "fourier_alpha": ("FLOAT", {"default": 1.00, "min": 0.1, "max": 4.0, "step": 0.1}),
                "perturb_mag_frac": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 0.05, "step": 0.001}),
                "enable_awb": ("BOOLEAN", {"default": True}),


                "enable_lut": ("BOOLEAN", {"default": True}),
                "lut": ("STRING", {"default": "X://insert/path/here(.png/.npy/.cube)", "vhs_path_extensions": lut_extensions}),
                "lut_strength": ("FLOAT", {"default": 1.00, "min": 0.0, "max": 1.0, "step": 0.01}),
                "glcm": ("BOOLEAN", {"default": False}),
                "glcm_distances": ("STRING", {"default": "1"}),
                "glcm_angles": ("STRING", {"default": f"0,{np.pi/4},{np.pi/2},{3*np.pi/4}"}),
                "glcm_levels": ("INT", {"default": 256, "min": 2, "max": 65536, "step": 1}),
                "glcm_strength": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "lbp": ("BOOLEAN", {"default": False}),
                "lbp_radius": ("INT", {"default": 3, "min": 1, "max": 50, "step": 1}),
                "lbp_n_points": ("INT", {"default": 24, "min": 1, "max": 512, "step": 1}),
                "lbp_method": (["default", "ror", "uniform", "var"], {"default": "uniform"}),
                "lbp_strength": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),

                # seed, exif
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**31-1, "step": 1}),
                "apply_exif_o": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "awb_ref_image": ("IMAGE",),
                "fft_ref_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("IMAGE", "EXIF")
    FUNCTION = "process"
    CATEGORY = "postprocessing"

    # default blocks for Camera and NS so the main node works even if user doesn't plug the helper nodes
    CAM_DEFAULTS = {
        "enable_bayer": True,
        "apply_jpeg_cycles_o": True,
        "jpeg_cycles": 1,
        "jpeg_quality": 88,
        "jpeg_qmax": 96,
        "apply_vignette_o": True,
        "vignette_strength": 0.35,
        "apply_chromatic_aberration_o": True,
        "ca_shift": 1.20,
        "iso_scale": 1.0,
        "read_noise": 2.0,
        "hot_pixel_prob": 1e-7,
        "apply_banding_o": True,
        "banding_strength": 0.0,
        "apply_motion_blur_o": True,
        "motion_blur_ksize": 1,
    }

    NS_DEFAULTS = {
        "non_semantic": False,
        "ns_iterations": 500,
        "ns_learning_rate": 3e-4,
        "ns_t_lpips": 4e-2,
        "ns_t_l2": 3e-5,
        "ns_c_lpips": 1e-2,
        "ns_c_l2": 0.6,
        "ns_grad_clip": 0.05,
    }

    def process(self, image,
                apply_noise_o=True,
                noise_std_frac=0.02,
                apply_clahe_o=True,
                clahe_clip=2.0,
                clahe_grid=8,
                fourier_cutoff=0.25,
                apply_fourier_o=True,
                fourier_strength=0.9,
                fourier_randomness=0.05,
                fourier_phase_perturb=0.08,
                fourier_radial_smooth=5,
                fourier_mode="auto",
                fourier_alpha=1.0,
                perturb_mag_frac=0.01,
                enable_awb=True,
                Cam_Opt="",
                NS_Opt="",
                enable_lut=True,
                lut="",
                lut_strength=1.0,
                glcm=False,
                glcm_distances="1",
                glcm_angles=f"0,{np.pi/4},{np.pi/2},{3*np.pi/4}",
                glcm_levels=256,
                glcm_strength=0.9,
                lbp=False,
                lbp_radius=3,
                lbp_n_points=24,
                lbp_method="uniform",
                lbp_strength=0.9,
                seed=-1,
                apply_exif_o=True,
                awb_ref_image=None,
                fft_ref_image=None
                ):

        if process_image is None:
            raise ImportError(f"Could not import process_image function: {IMPORT_ERROR}")

        # Parse Cam_Opt and NS_Opt JSON strings into dicts and merge with defaults
        cam_opts = dict(self.CAM_DEFAULTS)
        if isinstance(Cam_Opt, str) and Cam_Opt.strip() != "":
            try:
                loaded = json.loads(Cam_Opt)
                if isinstance(loaded, dict):
                    cam_opts.update(loaded)
            except Exception:
                pass

        ns_opts = dict(self.NS_DEFAULTS)
        if isinstance(NS_Opt, str) and NS_Opt.strip() != "":
            try:
                loaded = json.loads(NS_Opt)
                if isinstance(loaded, dict):
                    ns_opts.update(loaded)
            except Exception:
                pass

        tmp_files = []

        try:
            # ---- Input image -> temporary input file ----
            pil_img = to_pil_from_any(image[0])
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_input:
                input_path = tmp_input.name
                pil_img.save(input_path)
                tmp_files.append(input_path)

            # ---- AWB reference image if present ----
            awb_ref_path = None
            if awb_ref_image is not None:
                pil_ref_awb = to_pil_from_any(awb_ref_image[0])
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_ref_awb:
                    awb_ref_path = tmp_ref_awb.name
                    pil_ref_awb.save(awb_ref_path)
                    tmp_files.append(awb_ref_path)

            # ---- FFT reference image if present ----
            fft_ref_path = None
            if fft_ref_image is not None:
                pil_ref_fft = to_pil_from_any(fft_ref_image[0])
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_ref_fft:
                    fft_ref_path = tmp_ref_fft.name
                    pil_ref_fft.save(fft_ref_path)
                    tmp_files.append(fft_ref_path)

            # ---- Output path ----
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_output:
                output_path = tmp_output.name
                tmp_files.append(output_path)

            # Parse list-like UI inputs into native lists
            parsed_glcm_distances = _parse_int_list(glcm_distances)
            parsed_glcm_angles = _parse_float_list(glcm_angles)

            # Prepare args for process_image with updated keys (matches build_argparser())
            args = SimpleNamespace(
                # positional
                input=input_path,
                output=output_path,

                # AWB / refs
                awb=bool(enable_awb),
                ref=awb_ref_path,
                fft_ref=fft_ref_path,

                # basic corrections / noise / CLAHE
                noise_std=float(noise_std_frac),
                noise=bool(apply_noise_o),
                clahe=bool(apply_clahe_o),
                clahe_clip=float(clahe_clip),
                tile=int(clahe_grid),

                # Fourier / FFT matching
                fft=bool(apply_fourier_o),
                fstrength=float(fourier_strength) if apply_fourier_o else 0.0,
                randomness=float(fourier_randomness),
                seed=(None if int(seed) < 0 else int(seed)),
                fft_mode=str(fourier_mode),
                fft_alpha=float(fourier_alpha),
                phase_perturb=float(fourier_phase_perturb),
                radial_smooth=int(fourier_radial_smooth),
                cutoff=float(fourier_cutoff),

                # GLCM
                glcm=bool(glcm),
                glcm_distances=parsed_glcm_distances,
                glcm_angles=parsed_glcm_angles,
                glcm_levels=int(glcm_levels),
                glcm_strength=float(glcm_strength),

                # LBP
                lbp=bool(lbp),
                lbp_radius=int(lbp_radius),
                lbp_n_points=int(lbp_n_points),
                lbp_method=str(lbp_method),
                lbp_strength=float(lbp_strength),

                # Non-semantic attack (from ns_opts)
                non_semantic=bool(ns_opts.get("non_semantic", False)),
                ns_iterations=int(ns_opts.get("ns_iterations", 500)),
                ns_learning_rate=float(ns_opts.get("ns_learning_rate", 3e-4)),
                ns_t_lpips=float(ns_opts.get("ns_t_lpips", 4e-2)),
                ns_t_l2=float(ns_opts.get("ns_t_l2", 3e-5)),
                ns_c_lpips=float(ns_opts.get("ns_c_lpips", 1e-2)),
                ns_c_l2=float(ns_opts.get("ns_c_l2", 0.6)),
                ns_grad_clip=float(ns_opts.get("ns_grad_clip", 0.05)),

                # Camera simulator options (from cam_opts)
                sim_camera=True,
                no_no_bayer=not bool(cam_opts.get("enable_bayer", True)),
                jpeg_cycles=int(cam_opts.get("jpeg_cycles", 1)) if bool(cam_opts.get("apply_jpeg_cycles_o", True)) else 1,
                jpeg_qmin=int(cam_opts.get("jpeg_quality", 88)),
                jpeg_qmax=int(cam_opts.get("jpeg_qmax", 96)),
                vignette_strength=float(cam_opts.get("vignette_strength", 0.35)) if bool(cam_opts.get("apply_vignette_o", True)) else 0.0,
                chroma_strength=float(cam_opts.get("ca_shift", 1.20)) if bool(cam_opts.get("apply_chromatic_aberration_o", True)) else 0.0,
                iso_scale=float(cam_opts.get("iso_scale", 1.0)),
                read_noise=float(cam_opts.get("read_noise", 2.0)),
                hot_pixel_prob=float(cam_opts.get("hot_pixel_prob", 1e-7)),
                banding_strength=float(cam_opts.get("banding_strength", 0.0)) if bool(cam_opts.get("apply_banding_o", True)) else 0.0,
                motion_blur_kernel=int(cam_opts.get("motion_blur_ksize", 1)) if bool(cam_opts.get("apply_motion_blur_o", True)) else 1,

                # LUT
                lut=(lut if enable_lut and lut != "" else None),
                lut_strength=float(lut_strength),

                # utility flags (positive-style equivalents)
                perturb=(True if perturb_mag_frac > 0 else False),
                perturb_magnitude=float(perturb_mag_frac),
                blend=False
            )

            # ---- Run the processing function ----
            process_image(input_path, output_path, args)

            # ---- Load result (force RGB) ----
            output_img = Image.open(output_path).convert("RGB")
            img_out = np.array(output_img)

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
            img_float = img_out.astype(np.float32) / 255.0
            tensor_out = torch.from_numpy(img_float).to(dtype=torch.float32).unsqueeze(0)
            tensor_out = torch.clamp(tensor_out, 0.0, 1.0)

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
    "CameraOptionsNode": CameraOptionsNode,
    "NSOptionsNode": NSOptionsNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "NovaNodes": "Image Postprocess (NOVA NODES)",
    "CameraOptionsNode": "Camera Options (NOVA)",
    "NSOptionsNode": "Non-semantic Options (NOVA)",
}
