import json
from io import BytesIO
from typing import Any

import numpy as np
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError

from image_postprocess.processor import DEFAULT_STAGE_ORDER, build_processing_args, encode_image_array, process_array
from image_postprocess.utils import FOURIER_VARIANTS, load_lut_bytes

from .schemas import (
    AWBConfig,
    BlendConfig,
    ClaheConfig,
    FFTConfig,
    GLCMConfig,
    LBPConfig,
    LUTConfig,
    NoiseConfig,
    NonSemanticConfig,
    OutputFormat,
    PerturbConfig,
    PipelineConfig,
    SimCameraConfig,
    StageName,
)


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

STAGE_ROUTE_SEGMENTS = {
    'blend': 'blend',
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

REF_AWB_STAGES = {'awb'}
REF_FFT_STAGES = {'fft', 'glcm', 'lbp'}
LUT_STAGES = {'lut'}

app = FastAPI(
    title='Image Detection Bypass Utility API',
    version='1.0.0',
    description='FastAPI backend for running individual image post-processing stages and ordered multi-stage pipelines.',
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


async def _read_image(upload: UploadFile | None, field_name: str) -> np.ndarray | None:
    if upload is None:
        return None
    data = await upload.read()
    if not data:
        raise HTTPException(status_code=400, detail=f'{field_name} is empty')
    try:
        image = Image.open(BytesIO(data)).convert('RGB')
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail=f'{field_name} is not a valid image') from exc
    return np.array(image)


async def _read_lut(upload: UploadFile | None):
    if upload is None:
        return None, None
    data = await upload.read()
    if not data:
        raise HTTPException(status_code=400, detail='lut_file is empty')
    try:
        lut = load_lut_bytes(upload.filename or 'upload.lut', data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return lut, upload.filename or 'upload.lut'


def _validate_stage_config(stage_name: StageName, payload: dict[str, Any]):
    model_cls = STAGE_MODELS[stage_name]
    try:
        return model_cls.model_validate(payload).model_dump()
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=json.loads(exc.json())) from exc


def _validate_pipeline_order(config: PipelineConfig) -> list[StageName]:
    enabled_stages = list(config.stages.keys())
    if config.execution_order:
        if not config.stage_order:
            raise HTTPException(status_code=400, detail='stage_order is required when execution_order is true')
        if len(config.stage_order) != len(set(config.stage_order)):
            raise HTTPException(status_code=400, detail='stage_order cannot contain duplicates')
        missing = [stage for stage in enabled_stages if stage not in config.stage_order]
        extra = [stage for stage in config.stage_order if stage not in config.stages]
        if missing:
            raise HTTPException(status_code=400, detail=f'stage_order is missing enabled stages: {", ".join(missing)}')
        if extra:
            raise HTTPException(status_code=400, detail=f'stage_order contains stages not present in stages: {", ".join(extra)}')
        return list(config.stage_order)
    return [stage for stage in DEFAULT_STAGE_ORDER if stage in config.stages]


def _build_response(arr, output_format: OutputFormat, include_exif: bool, stage_order: list[str]):
    image_bytes, media_type = encode_image_array(arr, output_format=output_format, include_exif=include_exif)
    headers = {
        'X-Stage-Count': str(len(stage_order)),
        'X-Stage-Order': ','.join(stage_order),
    }
    return Response(content=image_bytes, media_type=media_type, headers=headers)


async def _run_stage_request(
    stage_name: StageName,
    config_raw: str | None,
    image: UploadFile,
    output_format: OutputFormat,
    include_exif: bool,
    ref_image: UploadFile | None = None,
    fft_ref_image: UploadFile | None = None,
    lut_file: UploadFile | None = None,
):
    config_model = _parse_json_config(config_raw, STAGE_MODELS[stage_name])
    input_arr = await _read_image(image, 'image')
    ref_arr_awb = await _read_image(ref_image, 'ref_image')
    ref_arr_fft = await _read_image(fft_ref_image, 'fft_ref_image')
    lut_data, lut_filename = await _read_lut(lut_file)

    if stage_name == 'lut' and lut_data is None:
        raise HTTPException(status_code=400, detail='lut_file is required for LUT processing')

    if stage_name == 'fft' and config_model.fft_variant not in FOURIER_VARIANTS:
        raise HTTPException(status_code=400, detail=f"Unsupported fft_variant '{config_model.fft_variant}'")

    args = build_processing_args(
        enabled_stages=[stage_name],
        stage_config=config_model.model_dump(),
        execution_order=True,
        stage_order=[stage_name],
        lut_path=lut_filename,
        lut_data=lut_data,
        include_exif=include_exif,
    )

    try:
        output_arr = process_array(input_arr, args, ref_arr_awb=ref_arr_awb, ref_arr_fft=ref_arr_fft)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return _build_response(output_arr, output_format, include_exif, [stage_name])


@app.get('/health', tags=['health'])
def health_check():
    return {'status': 'ok'}


@router.post('/process/pipeline')
async def process_pipeline(
    image: UploadFile = File(...),
    config: str = Form(...),
    output_format: OutputFormat = Form('png'),
    include_exif: bool = Form(True),
    ref_image: UploadFile | None = File(None),
    fft_ref_image: UploadFile | None = File(None),
    lut_file: UploadFile | None = File(None),
):
    pipeline_config = _parse_json_config(config, PipelineConfig)
    if not pipeline_config.stages:
        raise HTTPException(status_code=400, detail='stages must contain at least one enabled stage')

    stage_order = _validate_pipeline_order(pipeline_config)
    stage_configs = {stage_name: _validate_stage_config(stage_name, stage_payload) for stage_name, stage_payload in pipeline_config.stages.items()}
    input_arr = await _read_image(image, 'image')
    ref_arr_awb = await _read_image(ref_image, 'ref_image')
    ref_arr_fft = await _read_image(fft_ref_image, 'fft_ref_image')
    lut_data, lut_filename = await _read_lut(lut_file)

    if 'lut' in stage_configs and lut_data is None:
        raise HTTPException(status_code=400, detail='lut_file is required for LUT processing')

    fft_config = stage_configs.get('fft')
    if fft_config and fft_config['fft_variant'] not in FOURIER_VARIANTS:
        raise HTTPException(status_code=400, detail=f"Unsupported fft_variant '{fft_config['fft_variant']}'")

    merged_stage_config: dict[str, Any] = {}
    for stage_payload in stage_configs.values():
        merged_stage_config.update(stage_payload)

    args = build_processing_args(
        enabled_stages=list(stage_configs.keys()),
        stage_config=merged_stage_config,
        execution_order=pipeline_config.execution_order,
        stage_order=stage_order if pipeline_config.execution_order else None,
        lut_path=lut_filename,
        lut_data=lut_data,
        include_exif=include_exif,
    )

    try:
        output_arr = process_array(input_arr, args, ref_arr_awb=ref_arr_awb, ref_arr_fft=ref_arr_fft)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return _build_response(output_arr, output_format, include_exif, stage_order)


def _make_stage_endpoint(stage_name: StageName):
    async def _endpoint(
        image: UploadFile = File(...),
        config: str = Form('{}'),
        output_format: OutputFormat = Form('png'),
        include_exif: bool = Form(True),
        ref_image: UploadFile | None = File(None),
        fft_ref_image: UploadFile | None = File(None),
        lut_file: UploadFile | None = File(None),
    ):
        return await _run_stage_request(
            stage_name=stage_name,
            config_raw=config,
            image=image,
            output_format=output_format,
            include_exif=include_exif,
            ref_image=ref_image,
            fft_ref_image=fft_ref_image,
            lut_file=lut_file,
        )

    return _endpoint


for stage_name, route_segment in STAGE_ROUTE_SEGMENTS.items():
    _endpoint = _make_stage_endpoint(stage_name)

    router.add_api_route(f'/process/{route_segment}', _endpoint, methods=['POST'], name=f'process_{stage_name}')


app.include_router(router)