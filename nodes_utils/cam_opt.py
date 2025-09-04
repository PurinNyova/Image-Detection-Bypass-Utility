import json

class CameraOptionsNode:
    """
    Node that encapsulates camera simulation / JPEG / vignette / chromatic aberration / noise
    settings. Returns a JSON string that can be connected to the main NovaNodes node's "Cam_Opt" input.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "enable_bayer": ("BOOLEAN", {"default": True}),
                "apply_jpeg_cycles_o": ("BOOLEAN", {"default": True}),
                "jpeg_cycles": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "jpeg_quality": ("INT", {"default": 88, "min": 10, "max": 100, "step": 1}),
                "jpeg_qmax": ("INT", {"default": 96, "min": 10, "max": 100, "step": 1}),
                "apply_vignette_o": ("BOOLEAN", {"default": True}),
                "vignette_strength": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "apply_chromatic_aberration_o": ("BOOLEAN", {"default": True}),
                "ca_shift": ("FLOAT", {"default": 1.20, "min": 0.0, "max": 5.0, "step": 0.1}),
                "iso_scale": ("FLOAT", {"default": 1.00, "min": 0.1, "max": 16.0, "step": 0.1}),
                "read_noise": ("FLOAT", {"default": 2.00, "min": 0.0, "max": 50.0, "step": 0.1}),
                "hot_pixel_prob": ("FLOAT", {"default": 1e-7, "min": 0.0, "max": 1e-3, "step": 1e-7}),
                "apply_banding_o": ("BOOLEAN", {"default": True}),
                "banding_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "apply_motion_blur_o": ("BOOLEAN", {"default": True}),
                "motion_blur_ksize": ("INT", {"default": 1, "min": 1, "max": 31, "step": 2}),
            }
        }

    RETURN_TYPES = ("CAMERAOPT",)
    RETURN_NAMES = ("CAM_OPT",)
    FUNCTION = "get_cam_opts"
    CATEGORY = "postprocessing"

    def get_cam_opts(self,
                     enable_bayer=True,
                     apply_jpeg_cycles_o=True,
                     jpeg_cycles=1,
                     jpeg_quality=88,
                     jpeg_qmax=96,
                     apply_vignette_o=True,
                     vignette_strength=0.35,
                     apply_chromatic_aberration_o=True,
                     ca_shift=1.20,
                     iso_scale=1.0,
                     read_noise=2.0,
                     hot_pixel_prob=1e-7,
                     apply_banding_o=True,
                     banding_strength=0.0,
                     apply_motion_blur_o=True,
                     motion_blur_ksize=1,
                     ):
        cam_opts = {
            "enable_bayer": bool(enable_bayer),
            "apply_jpeg_cycles_o": bool(apply_jpeg_cycles_o),
            "jpeg_cycles": int(jpeg_cycles),
            "jpeg_quality": int(jpeg_quality),
            "jpeg_qmax": int(jpeg_qmax),
            "apply_vignette_o": bool(apply_vignette_o),
            "vignette_strength": float(vignette_strength),
            "apply_chromatic_aberration_o": bool(apply_chromatic_aberration_o),
            "ca_shift": float(ca_shift),
            "iso_scale": float(iso_scale),
            "read_noise": float(read_noise),
            "hot_pixel_prob": float(hot_pixel_prob),
            "apply_banding_o": bool(apply_banding_o),
            "banding_strength": float(banding_strength),
            "apply_motion_blur_o": bool(apply_motion_blur_o),
            "motion_blur_ksize": int(motion_blur_ksize),
        }
        return (json.dumps(cam_opts),)
