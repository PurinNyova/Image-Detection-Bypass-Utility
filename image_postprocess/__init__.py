__all__ = ["process_image"]


def __getattr__(name):
    if name == "process_image":
        from .processor import process_image
        return process_image
    raise AttributeError(name)