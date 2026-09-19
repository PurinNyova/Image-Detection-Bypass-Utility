"""Locked iPhone 16 Pro still-photo identity. Do not invent another body unless asked."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CameraProfile:
    make: str
    model: str
    software: str
    host_computer: str
    lens_make: str
    lens_model: str
    fnumber: float
    focal_mm: float
    focal_35: int
    white_balance: int  # 0 Auto
    metering_mode: int  # 5 Pattern/Multi-segment
    flash: int  # 16 Off, did not fire
    exposure_program: int  # 2 Program AE
    exposure_compensation: float
    color_space: int  # 1 sRGB


IPHONE_16_PRO = CameraProfile(
    make="Apple",
    model="iPhone 16 Pro",
    software="18.5",
    host_computer="iPhone 16 Pro",
    lens_make="Apple",
    lens_model="iPhone 16 Pro back triple camera 6.765mm f/1.78",
    fnumber=1.78,
    focal_mm=6.765,
    focal_35=24,
    white_balance=0,
    metering_mode=5,
    flash=16,
    exposure_program=2,
    exposure_compensation=0.0,
    color_space=1,
)

PROFILES = {"iphone_16_pro": IPHONE_16_PRO}


def get_profile(name: str) -> CameraProfile:
    key = (name or "iphone_16_pro").strip().lower().replace(" ", "_")
    if key not in PROFILES:
        raise KeyError(f"Unknown camera profile {name!r}. Known: {sorted(PROFILES)}")
    return PROFILES[key]
