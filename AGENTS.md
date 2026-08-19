# AGENTS

## Forensic camera JPEG

Before changing EXIF/JPEG encode/ELA: read [`image_postprocess/forensic_camera/README.md`](image_postprocess/forensic_camera/README.md). Do not re-derive. Do not upscale. Do not change camera body unless asked.

Entry: `apply_forensic_camera` from `image_postprocess.forensic_camera`, called by `process_image` when `forensic_camera=True` (default).
Donors: `image_postprocess/donors/`. ExifTool on PATH / `EXIFTOOL_PATH`.
