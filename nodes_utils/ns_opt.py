import json

class NSOptionsNode:
    """
    Node that encapsulates non-semantic attack parameters. Returns a JSON string
    that can be connected to the main NovaNodes node's "NS_Opt" input.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "non_semantic": ("BOOLEAN", {"default": False}),
                "ns_iterations": ("INT", {"default": 500, "min": 1, "max": 10000, "step": 1}),
                "ns_learning_rate": ("FLOAT", {"default": 3e-4, "min": 1e-6, "max": 1.0, "step": 1e-6}),
                "ns_t_lpips": ("FLOAT", {"default": 4e-2, "min": 0.0, "max": 1.0, "step": 1e-4}),
                "ns_t_l2": ("FLOAT", {"default": 3e-5, "min": 0.0, "max": 1.0, "step": 1e-6}),
                "ns_c_lpips": ("FLOAT", {"default": 1e-2, "min": 0.0, "max": 1.0, "step": 1e-4}),
                "ns_c_l2": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 10.0, "step": 1e-3}),
                "ns_grad_clip": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 1e-4}),
            }
        }

    RETURN_TYPES = ("NONSEMANTICOP",)
    RETURN_NAMES = ("NS_OPT",)
    FUNCTION = "get_ns_opts"
    CATEGORY = "postprocessing"

    def get_ns_opts(self,
                    non_semantic=False,
                    ns_iterations=500,
                    ns_learning_rate=3e-4,
                    ns_t_lpips=4e-2,
                    ns_t_l2=3e-5,
                    ns_c_lpips=1e-2,
                    ns_c_l2=0.6,
                    ns_grad_clip=0.05,
                    ):
        ns_opts = {
            "non_semantic": bool(non_semantic),
            "ns_iterations": int(ns_iterations),
            "ns_learning_rate": float(ns_learning_rate),
            "ns_t_lpips": float(ns_t_lpips),
            "ns_t_l2": float(ns_t_l2),
            "ns_c_lpips": float(ns_c_lpips),
            "ns_c_l2": float(ns_c_l2),
            "ns_grad_clip": float(ns_grad_clip),
        }
        return (json.dumps(ns_opts),)
