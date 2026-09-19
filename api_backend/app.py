import json
import logging
import os
import re
import tempfile
from datetime import datetime
from typing import Literal

import numpy as np
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError
from PIL.Image import DecompressionBombError
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.middleware.cors import CORSMiddleware

from image_postprocess.forensic_camera import ForensicOptions, apply_forensic_camera
from image_postprocess.forensic_camera.exiftool_bin import find_exiftool
from image_postprocess.processor import build_processing_args, encode_image_array, process_array
from image_postprocess.utils import load_lut_bytes

from .schemas import (
    AWBConfig,
    BlendConfig,
    ClaheConfig,
    FFTConfig,
    ForensicConfig,
    GLCMConfig,
    LBPConfig,
    LUTConfig,
    NoiseConfig,
    NonSemanticConfig,
    OutputFormat,
    PerturbConfig,
    SimCameraConfig,
    StageName,
)

logger = logging.getLogger('uvicorn.error')

MAX_UPLOAD_BYTES = 64 * 1024 * 1024
_UPLOAD_CHUNK = 1024 * 1024
_DATETIME_FORMATS = ('%Y:%m:%d %H:%M:%S', '%Y-%m-%d %H:%M:%S')

STAGE_MODELS = {
    'blend': BlendConfig,
    'non_semantic': NonSemanticConfig,
    'clahe': ClaheConfig,
    'fft': FFTConfig,
    'glcm': GLCMConfig,
    'lbp': LBPConfig,
    'noise': NoiseConfig,
    'perturb': PerturbConfig,
    'sim_camera': SimCameraConfig,
    'awb': AWBConfig,
    'lut': LUTConfig,
}

REF_AWB_STAGES = {'awb'}
REF_FFT_STAGES = {'fft', 'glcm', 'lbp'}
LUT_STAGES = {'lut'}

_PUBLIC_SEGMENT = {
    'blend': 'color-blend',
    'non_semantic': 'non-semantic',
    'clahe': 'clahe',
    'fft': 'fft',
    'glcm': 'glcm',
    'lbp': 'lbp',
    'noise': 'noise',
    'perturb': 'perturb',
    'sim_camera': 'sim-camera',
    'awb': 'awb',
    'lut': 'lut',
}

app = FastAPI(
    title='Image Detection Bypass Utility API',
    version='1.0.0',
    description='FastAPI backend for running individual image post-processing stages.',
)
router = APIRouter(prefix='/api/v1', tags=['processing'])


def _parse_json_config(config_raw: str | None, model_cls):
    try:
        payload = json.loads(config_raw) if config_raw else {}
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f'Invalid JSON in config: {exc.msg}') from exc
    try:
        return model_cls.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=json.loads(exc.json())) from exc


def _sanitize_name(filename: str | None, fallback: str) -> str:
    base = os.path.basename(filename or '').strip()
    ascii_safe = re.sub(r'[^A-Za-z0-9._-]+', '_', base)
    ascii_safe = ascii_safe.strip('._-') or fallback
    return ascii_safe[:64]


async def _read_bounded(upload: UploadFile, field_name: str) -> bytes:
    if upload is None:
        raise HTTPException(status_code=400, detail=f'{field_name} is required')
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(_UPLOAD_CHUNK)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f'{field_name} exceeds {MAX_UPLOAD_BYTES} byte upload cap')
        chunks.append(chunk)
    if not chunks:
        raise HTTPException(status_code=400, detail=f'{field_name} is empty')
    return b''.join(chunks)


def _decode_rgb(data: bytes, field_name: str) -> np.ndarray:
    from io import BytesIO
    try:
        image = Image.open(BytesIO(data)).convert('RGB')
    except (UnidentifiedImageError, OSError, DecompressionBombError) as exc:
        raise HTTPException(status_code=400, detail=f'{field_name} is not a valid image') from exc
    return np.array(image)


async def _reject_extras(request: Request, allowed: set[str]):
    form_keys = set((await request.form()).keys())
    extras = [key for key in form_keys if key not in allowed]
    if extras:
        raise HTTPException(status_code=422, detail=f'Unknown form field(s): {", ".join(sorted(extras))}')


async def _handle_stage(
    *, stage_name: StageName, request: Request,
    image: UploadFile, config_raw: str | None,
    output_format: OutputFormat, include_exif: bool,
    reference_image: UploadFile | None, lut_file: UploadFile | None,
    allowed_fields: set[str],
):
    await _reject_extras(request, allowed_fields)
    config_model = _parse_json_config(config_raw, STAGE_MODELS[stage_name])

    image_bytes = await _read_bounded(image, 'image')
    if lut_file is not None:
        lut_bytes = await _read_bounded(lut_file, 'lut_file')
    elif stage_name in LUT_STAGES:
        raise HTTPException(status_code=400, detail='lut_file is required for LUT processing')
    else:
        lut_bytes = None

    stage_config = config_model.model_dump()
    lut_name = None
    lut_data = None
    if stage_name in LUT_STAGES:
        lut_name = _sanitize_name(lut_file.filename, 'upload.lut')
        try:
            lut_data = await run_in_threadpool(load_lut_bytes, lut_name, lut_bytes)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    ref_arr_awb = None
    if stage_name in REF_AWB_STAGES and reference_image is not None:
        ref_arr_awb = await run_in_threadpool(
            _decode_rgb, await _read_bounded(reference_image, 'reference_image'), 'reference_image')
    ref_arr_fft = None
    if stage_name in REF_FFT_STAGES and reference_image is not None:
        ref_arr_fft = await run_in_threadpool(
            _decode_rgb, await _read_bounded(reference_image, 'reference_image'), 'reference_image')

    input_arr = await run_in_threadpool(_decode_rgb, image_bytes, 'image')

    args = build_processing_args(
        enabled_stages=[stage_name],
        stage_config=stage_config,
        execution_order=True,
        stage_order=[stage_name],
        lut_path=lut_name,
        lut_data=lut_data,
        include_exif=include_exif,
    )

    try:
        output_arr = await run_in_threadpool(
            process_array, input_arr, args,
            ref_arr_awb=ref_arr_awb, ref_arr_fft=ref_arr_fft, fail_fast=True,
        )
        image_out, media_type = await run_in_threadpool(
            encode_image_array, output_arr,
            output_format=output_format, include_exif=include_exif,
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception('Stage processing failed for %s', stage_name)
        raise HTTPException(status_code=500, detail='Internal processing error') from None

    return Response(
        content=image_out,
        media_type=media_type,
        headers={
            'X-Process': _PUBLIC_SEGMENT[stage_name],
            'X-Output-Format': output_format,
        },
    )


_BASE_FIELDS = {'image', 'config', 'output_format', 'include_exif'}


async def _call_stage(stage_name: StageName, request: Request, image: UploadFile, config: str | None,
                      output_format: OutputFormat, include_exif: bool,
                      reference_image: UploadFile | None, lut_file: UploadFile | None):
    allowed = set(_BASE_FIELDS)
    if stage_name in REF_AWB_STAGES or stage_name in REF_FFT_STAGES:
        allowed.add('reference_image')
    if stage_name in LUT_STAGES:
        allowed.add('lut_file')
    return await _handle_stage(
        stage_name=stage_name, request=request, image=image, config_raw=config,
        output_format=output_format, include_exif=include_exif,
        reference_image=reference_image, lut_file=lut_file,
        allowed_fields=allowed,
    )


def _make_plain_endpoint(stage_name: StageName):
    async def endpoint(request: Request, image: UploadFile = File(...), config: str | None = Form(None),
                       output_format: OutputFormat = Form('png'), include_exif: bool = Form(False)):
        return await _call_stage(stage_name, request, image, config, output_format, include_exif, None, None)
    return endpoint


def _make_ref_endpoint(stage_name: StageName):
    async def endpoint(request: Request, image: UploadFile = File(...), config: str | None = Form(None),
                       output_format: OutputFormat = Form('png'), include_exif: bool = Form(False),
                       reference_image: UploadFile | None = File(None)):
        return await _call_stage(stage_name, request, image, config, output_format, include_exif, reference_image, None)
    return endpoint


def _make_lut_endpoint():
    async def endpoint(request: Request, image: UploadFile = File(...), config: str | None = Form(None),
                       output_format: OutputFormat = Form('png'), include_exif: bool = Form(False),
                       lut_file: UploadFile | None = File(None)):
        return await _call_stage('lut', request, image, config, output_format, include_exif, None, lut_file)
    return endpoint


_STAGE_ENDPOINT_MAKERS = {
    'blend': _make_plain_endpoint,
    'non_semantic': _make_plain_endpoint,
    'clahe': _make_plain_endpoint,
    'noise': _make_plain_endpoint,
    'perturb': _make_plain_endpoint,
    'sim_camera': _make_plain_endpoint,
    'fft': _make_ref_endpoint,
    'glcm': _make_ref_endpoint,
    'lbp': _make_ref_endpoint,
    'awb': _make_ref_endpoint,
    'lut': lambda stage: _make_lut_endpoint(),
}


_FORENSIC_FIELDS = {'image', 'config'}


def _forensic_endpoint():
    async def endpoint(request: Request, image: UploadFile = File(...), config: str | None = Form(None)):
        await _reject_extras(request, _FORENSIC_FIELDS)
        cfg = _parse_json_config(config, ForensicConfig)

        dt = None
        if cfg.datetime_original:
            for fmt in _DATETIME_FORMATS:
                try:
                    dt = datetime.strptime(cfg.datetime_original.strip(), fmt)
                    break
                except ValueError:
                    continue
            if dt is None:
                raise HTTPException(
                    status_code=422,
                    detail="Invalid datetime_original; expected format 'YYYY-MM-DD HH:MM:SS' or 'YYYY:MM:DD HH:MM:SS'",
                )

        image_bytes = await _read_bounded(image, 'image')
        source_name = _sanitize_name(image.filename, 'image')

        def run_forensic():
            from io import BytesIO
            try:
                pil_image = Image.open(BytesIO(image_bytes))
            except (UnidentifiedImageError, OSError, DecompressionBombError) as exc:
                raise HTTPException(status_code=400, detail='image is not a valid image') from exc
            opts = ForensicOptions(
                profile=cfg.profile,
                software=cfg.software,
                datetime_original=dt,
                iso=cfg.iso,
                gps_lat=cfg.gps_lat,
                gps_lon=cfg.gps_lon,
                gps_alt=cfg.gps_alt,
                ela_flatten=cfg.ela_flatten,
                strip_fingerprints=cfg.strip_fingerprints,
                seed=cfg.seed,
                source_name=source_name,
            )
            with tempfile.TemporaryDirectory(prefix='forensic_') as tmpdir:
                dest_path = os.path.join(tmpdir, 'result.jpg')
                try:
                    out_path = apply_forensic_camera(pil_image, dest_path, opts)
                except Exception:
                    logger.exception('Forensic processing failed')
                    raise HTTPException(status_code=500, detail='Internal processing error') from None
                try:
                    with open(out_path, 'rb') as fh:
                        out_bytes = fh.read()
                finally:
                    try:
                        os.remove(out_path)
                    except OSError:
                        pass
            return out_bytes

        exiftool_available = bool(await run_in_threadpool(find_exiftool))

        try:
            out_bytes = await run_in_threadpool(run_forensic)
        except HTTPException:
            raise
        except Exception:
            logger.exception('Forensic processing failed')
            raise HTTPException(status_code=500, detail='Internal processing error') from None

        metadata_level = 'full' if exiftool_available else 'degraded'
        return Response(
            content=out_bytes,
            media_type='image/jpeg',
            headers={
                'X-Process': 'forensic-camera',
                'X-Output-Format': 'jpeg',
                'X-Forensic-Metadata': metadata_level,
            },
        )

    return endpoint


@app.get('/health', tags=['health'])
def health_check():
    return {'status': 'ok'}


for _stage, _segment in _PUBLIC_SEGMENT.items():
    _endpoint = _STAGE_ENDPOINT_MAKERS[_stage](_stage)
    router.add_api_route(f'/{_segment}', _endpoint, methods=['POST'], name=f'stage_{_stage}')

router.add_api_route('/forensic-camera', _forensic_endpoint(), methods=['POST'], name='forensic_camera')

app.include_router(router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        'http://localhost:3000',
        'http://127.0.0.1:3000',
    ],
    allow_methods=['GET', 'POST', 'OPTIONS'],
    allow_headers=['Content-Type'],
    expose_headers=['X-Process', 'X-Output-Format', 'X-Forensic-Metadata'],
)
