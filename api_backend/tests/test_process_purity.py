"""Purity tests for the processing pipeline and stage API.

Proves:
- one stage flag per endpoint (only the addressed stage flag is enabled),
- one-element stage_order is honored,
 - fault injection raises under fail_fast instead of returning unchanged output,
 - a failing process_array inside a stage endpoint becomes a generic 500,
 - a real stage endpoint's output matches one direct process_array+encode call,
- stage wrappers are equivalent to direct calls of the underlying functions,
- the ordinary (stage-endpoint) path never calls apply_forensic_camera,
- stage endpoints default include_exif to False.
"""

import argparse
import inspect
import io
import os
import numpy as np
import unittest
from importlib import import_module
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

from PIL import Image
from fastapi import HTTPException
from starlette.datastructures import UploadFile

app_module = import_module('api_backend.app')
from api_backend.app import _make_lut_endpoint, _make_plain_endpoint, _make_ref_endpoint
from image_postprocess import processor
from image_postprocess.processor import (
    DEFAULT_STAGE_ORDER,
    build_processing_args,
    encode_image_array,
    process_array,
    resolve_stage_order,
)


def tiny_png_bytes():
    buf = io.BytesIO()
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(buf, format='PNG')
    return buf.getvalue()


def make_upload():
    return UploadFile(file=io.BytesIO(tiny_png_bytes()), filename='a.png')


def make_lut_upload():
    lut = np.arange(256, dtype=np.uint8).repeat(3).reshape(256, 3)
    buf = io.BytesIO()
    np.save(buf, lut)
    buf.seek(0)
    return UploadFile(file=buf, filename='lut.npy')


def stub_request():
    async def form():
        return {}
    return SimpleNamespace(form=form)


class SingleStageFlagTests(unittest.TestCase):
    def test_enabled_stage_sets_only_its_flag(self):
        for stage in ('blend', 'noise', 'clahe', 'awb', 'sim_camera'):
            args = build_processing_args(enabled_stages=[stage])
            for name in DEFAULT_STAGE_ORDER:
                if name == 'lut':
                    continue
                self.assertEqual(getattr(args, name), name == stage, name)

    def test_lut_flag_only_via_lut_path(self):
        args = build_processing_args(enabled_stages=['lut'], lut_path='x.png')
        self.assertEqual(args.lut, 'x.png')
        self.assertIsNone(build_processing_args(enabled_stages=['lut']).lut)

    def test_one_element_stage_order_respected(self):
        for stage in ('noise', 'lbp', 'sim_camera'):
            args = build_processing_args(enabled_stages=[stage], execution_order=True,
                                         stage_order=[stage])
            self.assertEqual(resolve_stage_order(args), [stage])

    def test_explicit_stage_order_beats_raw_argv(self):
        args = build_processing_args(enabled_stages=['noise'], execution_order=True,
                                     stage_order=['noise'], raw_argv=['--noise', '--clahe'])
        self.assertEqual(resolve_stage_order(args), ['noise'])

    def test_execution_order_false_gives_full_pipeline(self):
        args = build_processing_args(enabled_stages=['noise'], execution_order=False,
                                     stage_order=['noise'])
        self.assertEqual(resolve_stage_order(args), list(DEFAULT_STAGE_ORDER))


class FailFastFaultInjectionTests(unittest.TestCase):
    ARR = np.full((4, 4, 3), 128, dtype=np.uint8)

    def noise_args(self):
        return build_processing_args(enabled_stages=['noise'], execution_order=True,
                                     stage_order=['noise'])

    def test_fail_fast_raises_on_stage_failure(self):
        with mock.patch.object(processor, 'add_gaussian_noise',
                               side_effect=RuntimeError('boom')):
            with self.assertRaises(RuntimeError):
                process_array(self.ARR, self.noise_args(), fail_fast=True)

    def test_without_fail_fast_returns_unchanged_output(self):
        args = build_processing_args(enabled_stages=['lut'], lut_path='nolink.cube',
                                     execution_order=True, stage_order=['lut'])
        with mock.patch.object(processor, 'load_lut', side_effect=ValueError('bad lut')):
            np.testing.assert_array_equal(
                process_array(self.ARR, args, fail_fast=False), self.ARR)

    def test_lut_fail_fast_raises(self):
        args = build_processing_args(enabled_stages=['lut'], lut_path='nolink.cube',
                                     execution_order=True, stage_order=['lut'])
        with mock.patch.object(processor, 'load_lut', side_effect=ValueError('bad lut')):
            with self.assertRaises(ValueError):
                process_array(self.ARR, args, fail_fast=True)
            np.testing.assert_array_equal(
                process_array(self.ARR, args, fail_fast=False), self.ARR)


class StageEquivalenceTests(unittest.TestCase):
    ARR = np.full((6, 6, 3), 100, dtype=np.uint8)

    def assert_wrapper_equals_direct(self, stage, func_name, marker, **extra):
        marker = np.asarray(marker, dtype=np.uint8)
        args = build_processing_args(enabled_stages=[stage], execution_order=True,
                                     stage_order=[stage], **extra)
        with mock.patch.object(processor, func_name, return_value=marker) as m:
            np.testing.assert_array_equal(process_array(self.ARR, args), marker)
        self.assertEqual(m.call_count, 1)
        return m

    def test_noise_wrapper_equals_direct_call(self):
        m = self.assert_wrapper_equals_direct('noise', 'add_gaussian_noise',
                                              np.zeros((6, 6, 3)))
        self.assertEqual(m.call_args.kwargs['std_frac'], 0.02)

    def test_clahe_wrapper_passes_config(self):
        m = self.assert_wrapper_equals_direct('clahe', 'clahe_color_correction',
                                              np.zeros((6, 6, 3)))
        self.assertEqual(m.call_args.kwargs['clip_limit'], 2.0)
        self.assertEqual(m.call_args.kwargs['tile_grid_size'], (8, 8))

    def test_perturb_wrapper_equals_direct_call(self):
        self.assert_wrapper_equals_direct('perturb', 'randomized_perturbation',
                                          np.zeros((6, 6, 3)))

    def test_disabled_stage_is_identity(self):
        for stage, func in (('noise', 'add_gaussian_noise'),
                            ('clahe', 'clahe_color_correction')):
            args = build_processing_args()
            with mock.patch.object(processor, func) as m:
                np.testing.assert_array_equal(process_array(self.ARR, args), self.ARR)
                m.assert_not_called()

    def test_seed_is_forwarded(self):
        args = build_processing_args(enabled_stages=['noise'], stage_config={'seed': 42},
                                     execution_order=True, stage_order=['noise'])
        with mock.patch.object(processor, 'add_gaussian_noise', return_value=self.ARR) as m:
            process_array(self.ARR, args)
        self.assertEqual(m.call_args.kwargs['seed'], 42)


class EndpointFailureAndEquivalenceTests(unittest.IsolatedAsyncioTestCase):
    def decode_rgb(self, data):
        return np.array(Image.open(io.BytesIO(data)).convert('RGB'))

    async def test_endpoint_wraps_stage_failure_in_generic_500(self):
        endpoint = _make_plain_endpoint('clahe')
        with mock.patch.object(app_module, 'process_array',
                               side_effect=RuntimeError('boom specific detail')):
            with self.assertRaises(HTTPException) as ctx:
                await endpoint(stub_request(), image=make_upload(), config=None,
                               output_format='png', include_exif=False)
        self.assertEqual(ctx.exception.status_code, 500)
        self.assertNotIn('boom', str(ctx.exception.detail))

    async def test_clahe_endpoint_output_equals_direct_process_call(self):
        endpoint = _make_plain_endpoint('clahe')
        resp = await endpoint(stub_request(), image=make_upload(), config=None,
                              output_format='png', include_exif=False)
        self.assertEqual(resp.status_code, 200)

        args = build_processing_args(enabled_stages=['clahe'], execution_order=True,
                                     stage_order=['clahe'], include_exif=False)
        direct_arr = process_array(self.decode_rgb(tiny_png_bytes()), args)
        direct_bytes, direct_media = encode_image_array(
            direct_arr, output_format='png', include_exif=False)
        self.assertEqual(resp.media_type, direct_media)
        self.assertEqual(resp.body, direct_bytes)

    async def test_clahe_endpoint_decoded_array_equals_direct_process_call(self):
        endpoint = _make_plain_endpoint('clahe')
        resp = await endpoint(stub_request(), image=make_upload(), config=None,
                              output_format='png', include_exif=False)
        endpoint_arr = self.decode_rgb(resp.body)
        args = build_processing_args(enabled_stages=['clahe'], execution_order=True,
                                     stage_order=['clahe'], include_exif=False)
        np.testing.assert_array_equal(
            endpoint_arr, process_array(self.decode_rgb(tiny_png_bytes()), args))


class OrdinaryPathForensicTests(unittest.IsolatedAsyncioTestCase):
    async def test_stage_endpoint_never_calls_apply_forensic_camera(self):
        for endpoint, extra in ((_make_plain_endpoint('clahe'), {}),
                                (_make_ref_endpoint('fft'), {'reference_image': None}),
                                (_make_lut_endpoint(), {'lut_file': make_lut_upload()})):
            with mock.patch.object(app_module, 'apply_forensic_camera') as forensic, \
                 mock.patch.object(app_module, 'process_array',
                                   return_value=np.zeros((8, 8, 3), dtype=np.uint8)), \
                 mock.patch.object(app_module, 'encode_image_array',
                                   return_value=(b'x', 'image/png')):
                resp = await endpoint(stub_request(), image=make_upload(), config=None,
                                      output_format='png', include_exif=False, **extra)
            self.assertEqual(resp.status_code, 200)
            forensic.assert_not_called()

    def test_process_image_without_flag_avoids_forensic(self):
        img_bytes = tiny_png_bytes()
        with TemporaryDirectory() as tmp:
            src = os.path.join(tmp, 'in.png')
            with open(src, 'wb') as fh:
                fh.write(img_bytes)
            dst = os.path.join(tmp, 'out.jpg')
            args = build_processing_args(include_exif=False)
            args.forensic_camera = False
            with mock.patch.object(processor, 'apply_forensic_camera') as forensic:
                processor.process_image(src, dst, args)
            forensic.assert_not_called()
            self.assertTrue(os.path.exists(dst))


class ExifDefaultsTests(unittest.IsolatedAsyncioTestCase):
    def test_endpoint_signatures_default_include_exif_false(self):
        for endpoint in (_make_plain_endpoint('clahe'), _make_ref_endpoint('fft'),
                         _make_lut_endpoint()):
            params = inspect.signature(endpoint).parameters
            self.assertEqual(params['include_exif'].default.default, False)

    async def test_endpoint_passes_include_exif_to_encoder(self):
        endpoint = _make_plain_endpoint('clahe')
        for value in (False, True):
            with mock.patch.object(app_module, 'process_array',
                                   return_value=np.zeros((8, 8, 3), dtype=np.uint8)), \
                 mock.patch.object(app_module, 'encode_image_array',
                                   return_value=(b'x', 'image/png')) as encoder:
                await endpoint(stub_request(), image=make_upload(), config=None,
                               output_format='png', include_exif=value)
            self.assertEqual(encoder.call_args.kwargs['include_exif'], value)

    def test_encode_image_array_omits_exif_when_disabled(self):
        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        png_bytes, media_type = encode_image_array(arr, output_format='png',
                                                   include_exif=False)
        self.assertEqual(media_type, 'image/png')
        with Image.open(io.BytesIO(png_bytes)) as img:
            self.assertIsNone(img.info.get('exif'))


if __name__ == '__main__':
    unittest.main()
