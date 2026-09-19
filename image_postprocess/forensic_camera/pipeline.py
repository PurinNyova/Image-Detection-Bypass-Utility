"""Final JPEG write: Apple DQT + iPhone EXIF/MakerNote + ELA flatten + fingerprint strip.

Pitfalls (do not rediscover):
- ExifTool cannot create MakerNotes from scratch → always -tagsFromFile donor HEIC first.
- Flash/Metering PrintConv fails unless: -n -ExifIFD:Flash=16 -ExifIFD:MeteringMode=5
- Copy ONLY -MakerNotes from donor, then overwrite Make/Model (donor is 13 Pro Max).
- After any ExifTool write, re-strip -JFIF:all= -XMP:all= (XMPToolkit reappears).
- Do not upscale to 48MP. ExifImageWidth/Height = actual pixels.
- GPSTimeStamp is UTC = DateTimeOriginal minus OffsetTime.
- Do not JPEG at Q95 then Apple DQT (double-compression 8x8 grid).
"""

from __future__ import annotations

import hashlib
import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps
from PIL.ExifTags import IFD
from PIL.TiffImagePlugin import IFDRational

from .exiftool_bin import find_exiftool, run_exiftool
from .profile import IPHONE_16_PRO, CameraProfile, get_profile

_DONORS = Path(__file__).resolve().parents[1] / "donors"
DONOR_DQT = _DONORS / "iphone_hdr_YES.jpg"
DONOR_MAKERNOTE = _DONORS / "iphone_13_pro_max.HEIC"

# EXIF tag ids
T_MAKE, T_MODEL, T_ORIENTATION = 0x010F, 0x0110, 0x0112
T_XRES, T_YRES, T_RESUNIT = 0x011A, 0x011B, 0x0128
T_SOFTWARE, T_DATETIME, T_YCBCRPOS = 0x0131, 0x0132, 0x0213
T_EXPOSURETIME, T_FNUMBER, T_EXPOSUREPROGRAM = 0x829A, 0x829D, 0x8822
T_ISO, T_EXIFVERSION = 0x8827, 0x9000
T_DATETIMEORIGINAL, T_DATETIMEDIGITIZED = 0x9003, 0x9004
T_OFFSETTIME, T_OFFSETTIMEORIGINAL, T_OFFSETTIMEDIGITIZED = 0x9010, 0x9011, 0x9012
T_COMPONENTS, T_SHUTTER, T_APERTURE = 0x9101, 0x9201, 0x9202
T_EXPOSUREBIAS, T_METERING, T_FLASH, T_FOCALLENGTH = 0x9204, 0x9207, 0x9209, 0x920A
T_SUBSEC, T_SUBSECORIG, T_SUBSECDIG = 0x9290, 0x9291, 0x9292
T_FLASHPIX, T_COLORSPACE = 0xA000, 0xA001
T_EXIFWIDTH, T_EXIFHEIGHT = 0xA002, 0xA003
T_SENSING, T_SCENE, T_WHITEBALANCE = 0xA217, 0xA301, 0xA403
T_FOCAL35, T_SCENECAPTURE = 0xA405, 0xA406
T_LENSMAKE, T_LENSMODEL = 0xA433, 0xA434


@dataclass
class ForensicOptions:
    profile: str = "iphone_16_pro"
    software: str | None = None
    datetime_original: datetime | None = None
    offset: str | None = None  # e.g. "+01:00"; default from local tz
    iso: int | None = None
    exposure_num: int = 1
    exposure_den: int | None = None
    gps_lat: float | None = None
    gps_lon: float | None = None
    gps_alt: float | None = None
    ela_flatten: bool = True
    ela_blur: float = 0.42
    ela_noise: float = 0.65
    strip_fingerprints: bool = True
    seed: int | None = None
    source_name: str = "image"


def _seed(opts: ForensicOptions) -> int:
    if opts.seed is not None:
        return int(opts.seed)
    return int(hashlib.sha1(opts.source_name.encode()).hexdigest()[:8], 16)


def _load_qtables() -> dict[int, list[int]]:
    if not DONOR_DQT.is_file():
        raise FileNotFoundError(f"Missing DQT donor: {DONOR_DQT}")
    with Image.open(DONOR_DQT) as donor:
        if not donor.quantization:
            raise RuntimeError("DQT donor has no JPEG quantization tables")
        return {k: list(v) for k, v in donor.quantization.items()}


def _uuid(name: str, salt: str) -> str:
    digest = hashlib.sha1(f"{salt}:{name}".encode()).hexdigest()
    return str(uuid.UUID(digest[:32])).upper()


def _accel(name: str) -> str:
    d = hashlib.sha1(name.encode()).digest()
    x = (d[0] - 127) / 1270.0
    y = (d[1] - 127) / 1270.0
    z = -0.972 - (d[2] / 2550.0)
    return f"{x:.6f} {y:.6f} {z:.6f}"


def _homogenize(im: Image.Image, opts: ForensicOptions) -> Image.Image:
    im = ImageOps.exif_transpose(im).convert("RGB")
    if opts.ela_blur > 0:
        im = im.filter(ImageFilter.GaussianBlur(radius=opts.ela_blur))
    if opts.ela_noise <= 0:
        return im
    arr = np.asarray(im, dtype=np.float32)
    rng = np.random.default_rng(_seed(opts))
    arr += rng.normal(0.0, opts.ela_noise, arr.shape).astype(np.float32)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


def _thumb_bytes(im: Image.Image, qtables: dict) -> bytes:
    thumb = ImageOps.contain(im, (320, 320)).convert("RGB")
    buf = BytesIO()
    thumb.save(buf, format="JPEG", qtables=qtables, subsampling="4:2:0")
    return buf.getvalue()


def _parse_offset(s: str) -> timedelta:
    sign = 1 if s[0] != "-" else -1
    body = s[1:] if s[0] in "+-" else s
    hh, mm = body.split(":")
    return sign * timedelta(hours=int(hh), minutes=int(mm))


def _local_offset(dt: datetime) -> str:
    aware = dt.astimezone() if dt.tzinfo else dt.replace(tzinfo=datetime.now().astimezone().tzinfo)
    off = aware.utcoffset() or timedelta(0)
    total = int(off.total_seconds())
    sign = "+" if total >= 0 else "-"
    total = abs(total)
    return f"{sign}{total // 3600:02d}:{(total % 3600) // 60:02d}"


def _build_pillow_exif(im: Image.Image, profile: CameraProfile, opts: ForensicOptions) -> Image.Exif:
    dt = opts.datetime_original or datetime.now()
    if dt.tzinfo:
        dt = dt.replace(tzinfo=None)
    offset = opts.offset or _local_offset(opts.datetime_original or datetime.now())
    iso = opts.iso or 80
    exp_n = opts.exposure_num
    exp_d = opts.exposure_den or 120
    stamp = dt.strftime("%Y:%m:%d %H:%M:%S")
    subsec = f"{dt.microsecond // 1000:03d}"
    software = opts.software or profile.software
    w, h = im.size

    exif = Image.Exif()
    exif[T_MAKE] = profile.make
    exif[T_MODEL] = profile.model
    exif[T_ORIENTATION] = 1
    exif[T_XRES] = IFDRational(72, 1)
    exif[T_YRES] = IFDRational(72, 1)
    exif[T_RESUNIT] = 2
    exif[T_SOFTWARE] = software
    exif[T_DATETIME] = stamp
    exif[T_YCBCRPOS] = 1
    ifd = exif.get_ifd(IFD.Exif)
    ifd[T_EXPOSURETIME] = IFDRational(exp_n, exp_d)
    ifd[T_FNUMBER] = IFDRational(int(round(profile.fnumber * 100)), 100)
    ifd[T_EXPOSUREPROGRAM] = profile.exposure_program
    ifd[T_ISO] = iso
    ifd[T_EXIFVERSION] = b"0232"
    ifd[T_DATETIMEORIGINAL] = stamp
    ifd[T_DATETIMEDIGITIZED] = stamp
    ifd[T_OFFSETTIME] = offset
    ifd[T_OFFSETTIMEORIGINAL] = offset
    ifd[T_OFFSETTIMEDIGITIZED] = offset
    ifd[T_COMPONENTS] = b"\x01\x02\x03\x00"
    seconds = exp_n / exp_d
    ifd[T_SHUTTER] = IFDRational(round(-math.log2(seconds) * 1_000_000), 1_000_000)
    ifd[T_APERTURE] = IFDRational(round(2 * math.log2(profile.fnumber) * 1_000_000), 1_000_000)
    ifd[T_EXPOSUREBIAS] = IFDRational(0, 1)
    ifd[T_METERING] = profile.metering_mode
    ifd[T_FLASH] = profile.flash
    ifd[T_FOCALLENGTH] = IFDRational(int(round(profile.focal_mm * 1000)), 1000)
    ifd[T_SUBSEC] = subsec
    ifd[T_SUBSECORIG] = subsec
    ifd[T_SUBSECDIG] = subsec
    ifd[T_FLASHPIX] = b"0100"
    ifd[T_COLORSPACE] = profile.color_space
    ifd[T_EXIFWIDTH] = w
    ifd[T_EXIFHEIGHT] = h
    ifd[T_SENSING] = 2
    ifd[T_SCENE] = 1
    ifd[T_WHITEBALANCE] = profile.white_balance
    ifd[T_FOCAL35] = profile.focal_35
    ifd[T_SCENECAPTURE] = 0
    ifd[T_LENSMAKE] = profile.lens_make
    ifd[T_LENSMODEL] = profile.lens_model
    return exif


def _stamp_exiftool(path: Path, im: Image.Image, profile: CameraProfile, opts: ForensicOptions, qtables: dict) -> None:
    dt = opts.datetime_original or datetime.now()
    naive = dt.replace(tzinfo=None) if dt.tzinfo else dt
    offset = opts.offset or _local_offset(dt)
    stamp = naive.strftime("%Y:%m:%d %H:%M:%S")
    subsec = f"{naive.microsecond // 1000:03d}"
    software = opts.software or profile.software
    iso = opts.iso or 80
    exp_n = opts.exposure_num
    exp_d = opts.exposure_den or 120
    w, h = im.size
    name = opts.source_name

    if DONOR_MAKERNOTE.is_file():
        run_exiftool(["-tagsFromFile", str(DONOR_MAKERNOTE), "-MakerNotes", str(path)])

    args = [
        f"-Make={profile.make}",
        f"-Model={profile.model}",
        f"-LensMake={profile.lens_make}",
        f"-LensModel={profile.lens_model}",
        f"-Software={software}",
        f"-HostComputer={profile.host_computer}",
        f"-DateTimeOriginal={stamp}",
        f"-CreateDate={stamp}",
        f"-ModifyDate={stamp}",
        f"-OffsetTime={offset}",
        f"-OffsetTimeOriginal={offset}",
        f"-OffsetTimeDigitized={offset}",
        f"-SubSecTimeOriginal={subsec}",
        f"-SubSecTimeDigitized={subsec}",
        f"-SubSecTime={subsec}",
        f"-ISO={iso}",
        f"-FNumber={profile.fnumber}",
        f"-ExposureTime={exp_n}/{exp_d}",
        f"-FocalLength={profile.focal_mm}",
        f"-FocalLengthIn35mmFormat={profile.focal_35}",
        "-WhiteBalance=Auto",
        "-ExposureProgram=Program AE",
        "-ExposureCompensation=0",
        "-ColorSpace=sRGB",
        f"-ExifImageWidth={w}",
        f"-ExifImageHeight={h}",
        f"-Apple:PhotoIdentifier={_uuid(name, 'photo')}",
        f"-Apple:ImageUniqueID={_uuid(name, 'content')}",
        f"-Apple:AccelerationVector={_accel(name)}",
        "-Apple:CameraType=Back Normal",
        "-Apple:ImageCaptureType=Photo",
    ]
    if opts.gps_lat is not None and opts.gps_lon is not None:
        lat, lon = opts.gps_lat, opts.gps_lon
        lat_ref = "N" if lat >= 0 else "S"
        lon_ref = "E" if lon >= 0 else "W"
        utc = naive - _parse_offset(offset)
        args += [
            f"-GPSLatitude={abs(lat):.7f}",
            f"-GPSLatitudeRef={lat_ref}",
            f"-GPSLongitude={abs(lon):.7f}",
            f"-GPSLongitudeRef={lon_ref}",
            f"-GPSAltitude={opts.gps_alt if opts.gps_alt is not None else 0:.1f}",
            "-GPSAltitudeRef=Above Sea Level" if (opts.gps_alt or 0) >= 0 else "-GPSAltitudeRef=Below Sea Level",
            f"-GPSDateStamp={utc.strftime('%Y:%m:%d')}",
            f"-GPSTimeStamp={utc.strftime('%H:%M:%S')}",
        ]
    thumb = path.with_suffix(".thumb.jpg")
    thumb.write_bytes(_thumb_bytes(im, qtables))
    args += [f"-ThumbnailImage<={thumb}", str(path)]
    try:
        run_exiftool(args)
        run_exiftool(
            [
                "-n",
                f"-ExifIFD:MeteringMode={profile.metering_mode}",
                f"-ExifIFD:Flash={profile.flash}",
                str(path),
            ]
        )
    finally:
        thumb.unlink(missing_ok=True)


def apply_forensic_camera(image: Image.Image, dest_path: str | Path, opts: ForensicOptions | None = None) -> Path:
    """Encode `image` as a camera-like JPEG at dest_path. Always .jpg."""
    opts = opts or ForensicOptions()
    profile = get_profile(opts.profile)
    qtables = _load_qtables()
    dest = Path(dest_path)
    if dest.suffix.lower() not in {".jpg", ".jpeg"}:
        dest = dest.with_suffix(".jpg")

    im = image.convert("RGB")
    if opts.ela_flatten:
        im = _homogenize(im, opts)

    rng = np.random.default_rng(_seed(opts))
    if opts.iso is None:
        opts.iso = int(rng.choice([50, 64, 80, 100]))
    if opts.exposure_den is None:
        opts.exposure_den = int(rng.integers(80, 124))

    exif = _build_pillow_exif(im, profile, opts)
    dest.parent.mkdir(parents=True, exist_ok=True)
    im.save(dest, format="JPEG", qtables=qtables, subsampling="4:2:0", exif=exif)

    if find_exiftool():
        _stamp_exiftool(dest, im, profile, opts, qtables)
        if opts.strip_fingerprints:
            run_exiftool(["-JFIF:all=", "-XMP:all=", str(dest)])
    elif opts.strip_fingerprints:
        # No ExifTool: at least drop Pillow JFIF by rewriting? Skip; piexif path keeps JFIF.
        pass
    return dest
