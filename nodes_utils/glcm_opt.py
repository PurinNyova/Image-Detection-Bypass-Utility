import json
import numpy as np


class GLCMOptionsNode:
    """
    Node that encapsulates GLCM normalization settings. Returns a JSON string
    that can be connected to the main NovaNodes node's "GLCM_Opt" input.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "glcm": ("BOOLEAN", {"default": False}),
                "glcm_distances": ("STRING", {"default": "1"}),
                "glcm_angles": ("STRING", {"default": f"0,{np.pi/4},{np.pi/2},{3*np.pi/4}"}),
                "glcm_levels": ("INT", {"default": 256, "min": 2, "max": 65536, "step": 1}),
                "glcm_strength": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("GLCMOPT",)
    RETURN_NAMES = ("GLCM_OPT",)
    FUNCTION = "get_glcm_opts"
    CATEGORY = "postprocessing"

    def get_glcm_opts(self,
                      glcm=False,
                      glcm_distances="1",
                      glcm_angles=f"0,{np.pi/4},{np.pi/2},{3*np.pi/4}",
                      glcm_levels=256,
                      glcm_strength=0.9,
                      ):
        glcm_opts = {
            "glcm": bool(glcm),
            "glcm_distances": str(glcm_distances),
            "glcm_angles": str(glcm_angles),
            "glcm_levels": int(glcm_levels),
            "glcm_strength": float(glcm_strength),
        }
        return (json.dumps(glcm_opts),)