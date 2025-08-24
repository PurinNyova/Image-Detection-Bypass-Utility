from PIL import Image

def remove_exif_pil(img: Image.Image) -> Image.Image:
    data = img.tobytes()
    new = Image.frombytes(img.mode, img.size, data)
    return new