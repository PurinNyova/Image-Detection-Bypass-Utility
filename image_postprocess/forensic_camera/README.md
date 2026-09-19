# Forensic camera pipeline (LLM playbook)

Last JPEG write. Replaces the piexif `PurinCamera` stub.

## What it writes
iPhone 16 Pro still: Make/Model/LensModel/HostComputer, Software=`18.5` (version only), Apple MakerNote (from donor HEIC), shooting EXIF, optional GPS, IFD1 thumbnail, Apple DQT 4:2:0, ELA flatten, strip JFIF/XMP/C2PA/AIGC.

## Files
```
image_postprocess/forensic_camera/   # this package
image_postprocess/donors/iphone_13_pro_max.HEIC   # MakerNote donor (do not copy Make/Model)
image_postprocess/donors/iphone_hdr_YES.jpg       # Apple quantization tables
```

ExifTool binary is **not** vendored. `PATH` / `EXIFTOOL_PATH` / `image_postprocess/bin/ExifTool.exe`. Without it: Pillow EXIF only, **no MakerNote**.

## Call
`apply_forensic_camera(pil_image, dest_path, ForensicOptions(...))` from `process_image` when `args.forensic_camera` (default on).

CLI: `--forensic-camera` (default) `--forensic-software 18.5` `--gps-lat --gps-lon --gps-alt` `--ela-flatten` `--strip-fingerprints` `--no-forensic-camera`.

## Order
1. Optional global GaussianBlur+noise (ELA flatten; whole frame, never local paint).
2. One JPEG encode with donor qtables, subsampling 4:2:0.
3. `exiftool -tagsFromFile DONOR.HEIC -MakerNotes` then overwrite identity/shooting/GPS.
4. Thumbnail `<=` 320px JPEG.
5. `-n -ExifIFD:Flash=16 -ExifIFD:MeteringMode=5`
6. `-JFIF:all= -XMP:all=`

## Pitfalls
1. Cannot create MakerNotes from scratch.
2. Flash/Metering PrintConv fails unless numeric `-n -ExifIFD:...`.
3. Donor is 13 Pro Max — copy **only** MakerNotes.
4. ExifTool re-adds `XMPToolkit` → always re-strip XMP.
5. Do not upscale to 48MP. ExifImage* = actual pixels.
6. GPSTimeStamp is UTC.
7. Do not Q95 then Apple DQT (double-compression grid).
8. `Software` must never be PIL/Pillow/OpenCV/Automatic1111/ComfyUI.

## Forbidden (must be absent)
C2PA/JUMBF, `DigitalSourceType=trainedAlgorithmicMedia`, AISystemUsed/prompt/seed/sampler/CFG, AIGC JSON comments.

## Verify
```
exiftool -G1 -s -Make -Model -LensModel -Software -DateTimeOriginal -ISO -FNumber \
  -Flash -MeteringMode -GPSLatitude -Apple:MakerNoteVersion -XMP:all -JFIF:all FILE
```
Pillow `quantization` == donor JPEG tables. ELA vs Q95: mean ~0.5–0.7, no paste rectangles.
