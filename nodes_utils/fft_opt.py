import json


class FFTOptionsNode:
    """
    Node that encapsulates FFT spectral matching settings. Returns a JSON string
    that can be connected to the main NovaNodes node's "FFT_Opt" input.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "apply_fourier_o": ("BOOLEAN", {"default": True}),
                "fourier_cutoff": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "fourier_strength": ("FLOAT", {"default": 0.90, "min": 0.0, "max": 1.0, "step": 0.01}),
                "fourier_randomness": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.5, "step": 0.01}),
                "fourier_phase_perturb": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 0.5, "step": 0.01}),
                "fourier_radial_smooth": ("INT", {"default": 5, "min": 0, "max": 50, "step": 1}),
                "fourier_mode": (["auto", "ref", "model"], {"default": "auto"}),
                "fourier_alpha": ("FLOAT", {"default": 1.00, "min": 0.1, "max": 4.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("FFTOPT",)
    RETURN_NAMES = ("FFT_OPT",)
    FUNCTION = "get_fft_opts"
    CATEGORY = "postprocessing"

    def get_fft_opts(self,
                     apply_fourier_o=True,
                     fourier_cutoff=0.25,
                     fourier_strength=0.9,
                     fourier_randomness=0.05,
                     fourier_phase_perturb=0.08,
                     fourier_radial_smooth=5,
                     fourier_mode="auto",
                     fourier_alpha=1.0,
                     ):
        fft_opts = {
            "apply_fourier_o": bool(apply_fourier_o),
            "fourier_cutoff": float(fourier_cutoff),
            "fourier_strength": float(fourier_strength),
            "fourier_randomness": float(fourier_randomness),
            "fourier_phase_perturb": float(fourier_phase_perturb),
            "fourier_radial_smooth": int(fourier_radial_smooth),
            "fourier_mode": str(fourier_mode),
            "fourier_alpha": float(fourier_alpha),
        }
        return (json.dumps(fft_opts),)