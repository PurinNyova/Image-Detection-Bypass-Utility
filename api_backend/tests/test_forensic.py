"""Forensic camera endpoint tests (standard library only).

Covers: JPEG-only output + expected headers, temporary directory cleanup on
success and on raised failure, degraded (no ExifTool) header, source name
sanitization, and that ordinary stage transforms never invoke the forensic
pipeline.
"""

import io
import os
import tempfile
import unittest
from unittest import mock

from PIL import Image

# `api_backend.app` is shadowed by the FastAPI instance bound as package attr;
# go through importlib to get the real module.
from importlib import import_module

# Pillow does not re-export DecompressionBombError at top level; shim for app import.
import PIL
from PIL.Image import DecompressionBombError as _DecompressionBombError

if not hasattr(PIL, 'DecompressionBombError'):
    PIL.DecompressionBombError = _DecompressionBombError

app_module = import_module('api_backend.app')
from image_postprocess.forensic_camera import apply_forensic_camera
from fastapi import HTTPException


def _jpeg_bytes(size=(8, 8)) -> bytes:
    buf = io.BytesIO()
    Image.new('RGB', size, (128, 64, 32)).save(buf, format='JPEG')
    return buf.getvalue()


def _fake_upload(data: bytes, filename: str = 'weird ../na*me!.PNG'):
    class Upload:
        _read_done = False

        async def read(self, _n=0):
            if not self._read_done:
                self._read_done = True
                return data
            return b''

    upload = Upload()
    upload.filename = filename
    return upload


class _RecordingTmpdir:
    """Delegates to the real TemporaryDirectory while recording the path."""

    instances: list[tempfile.TemporaryDirectory] = []
    _real_cls = tempfile.TemporaryDirectory

    def __init__(self, *args, **kwargs):
        self._real = _RecordingTmpdir._real_cls(*args, **kwargs)
        _RecordingTmpdir.instances.append(self)

    def __getattr__(self, name):
        return getattr(self._real, name)

    def __enter__(self):
        return self._real.__enter__()

    def __exit__(self, *args):
        return self._real.__exit__(*args)


class ForensicEndpointTests(unittest.IsolatedAsyncioTestCase):

    def _endpoint(self):
        return app_module._forensic_endpoint()

    async def _invoke(self, edp, image_bytes, filename='my ../syn*th.photo.PXN', config=None):
        request = mock.MagicMock()
        request._form = None
        request.form = mock.AsyncMock(return_value={})
        return await edp(
            request,
            image=_fake_upload(image_bytes, filename),
            config=config,
        )

    def _patch_tempdir(self):
        _RecordingTmpdir.instances.clear()
        return mock.patch.object(app_module.tempfile, 'TemporaryDirectory', _RecordingTmpdir)

    async def test_jpeg_only_output_and_headers_degraded(self):
        edp = self._endpoint()
        with mock.patch.object(app_module, 'find_exiftool', return_value=None):
            response = await self._invoke(edp, _jpeg_bytes(), filename='camera roll.JPG')
        self.assertIsNotNone(response)
        self.assertEqual(response.media_type, 'image/jpeg')
        self.assertTrue(response.body.startswith(b'\xff\xd8\xff'), 'JPEG magic missing')
        with Image.open(io.BytesIO(response.body)) as im:
            self.assertEqual(im.format, 'JPEG')
        self.assertEqual(response.headers['X-Process'], 'forensic-camera')
        self.assertEqual(response.headers['X-Output-Format'], 'jpeg')
        self.assertEqual(response.headers['X-Forensic-Metadata'], 'degraded')

    async def test_full_header_when_exiftool_forced_present(self):
        edp = self._endpoint()

        def fake_apply(image, dest_path, opts):
            out = str(dest_path)
            with open(out, 'wb') as fh:
                fh.write(b'\xff\xd8\xffE0\x00\x02\x10\x10\x00\x00')
            return out

        with mock.patch.object(app_module, 'find_exiftool', return_value='exiftool'), \
                mock.patch.object(app_module, 'apply_forensic_camera', fake_apply), \
                self._patch_tempdir():
            response = await self._invoke(edp, _jpeg_bytes())
        self.assertEqual(response.media_type, 'image/jpeg')
        self.assertEqual(response.headers['X-Forensic-Metadata'], 'full')

    async def test_jpeg_only_even_for_non_jpeg_source_name(self):
        edp = self._endpoint()
        response = await self._invoke(edp, _jpeg_bytes(), filename='input.png')
        with Image.open(io.BytesIO(response.body)) as im:
            self.assertEqual(im.format, 'JPEG')

    async def test_tempdir_removed_on_success(self):
        edp = self._endpoint()
        seen = {}

        def fake_apply(image, dest_path, opts):
            seen['parent'] = os.path.abspath(os.path.dirname(str(dest_path)))
            seen['opts_source_name'] = getattr(opts, 'source_name', None)
            out = str(dest_path)
            with open(out, 'wb') as fh:
                fh.write(b'\xff\xd8\xffE0\x00\x02\x10\x10\x00\x00')
            return out

        with mock.patch.object(app_module, 'apply_forensic_camera', fake_apply), \
                self._patch_tempdir():
            response = await self._invoke(edp, _jpeg_bytes())

        self.assertEqual(response.media_type, 'image/jpeg')
        self.assertEqual(len(_RecordingTmpdir.instances), 1)
        tmpdir_name = _RecordingTmpdir.instances[0].name
        self.assertEqual(seen['parent'], os.path.abspath(tmpdir_name))
        self.assertFalse(os.path.exists(tmpdir_name), 'temp directory leaked on success')

    async def test_tempdir_removed_on_raised_failure(self):
        edp = self._endpoint()

        def failing_apply(image, dest_path, opts):
            raise RuntimeError('boom')

        with mock.patch.object(app_module, 'apply_forensic_camera', failing_apply), \
                self._patch_tempdir():
            with self.assertRaises(HTTPException) as ctx:
                await self._invoke(edp, _jpeg_bytes())
        tmpdir_name = _RecordingTmpdir.instances[0].name
        self.assertFalse(os.path.exists(tmpdir_name), 'temp directory leaked on failure')
        self.assertEqual(ctx.exception.status_code, 500)

    async def test_source_name_sanitized_into_options(self):
        edp = self._endpoint()
        seen = {}

        def fake_apply(image, dest_path, opts):
            seen['source_name'] = opts.source_name
            out = str(dest_path)
            with open(out, 'wb') as fh:
                fh.write(b'\xff\xd8\xffE0\x00\x02\x10\x10\x00\x00')
            return out

        with mock.patch.object(app_module, 'apply_forensic_camera', fake_apply), \
                self._patch_tempdir():
            await self._invoke(
                edp, _jpeg_bytes(),
                filename='../../etc/passwd with spaces & ünïcode.tiff',
            )
        source_name = seen['source_name']
        self.assertTrue(source_name)
        self.assertLessEqual(len(source_name), 64)
        forbidden = set(os.sep) | {'/', '\\', '*', '?', '"', '<', '>', '|', ':'}
        self.assertEqual(forbidden & set(source_name), set())
        self.assertRegex(source_name, r'^[A-Za-z0-9._-]+$')


class OrdinaryTransformNeverForensicTests(unittest.IsolatedAsyncioTestCase):

    async def test_noise_stage_never_calls_forensic(self):
        endpoint = app_module._make_plain_endpoint('noise')
        request = mock.MagicMock()
        request._form = None
        request.form = mock.AsyncMock(return_value={})
        with mock.patch.object(
            app_module, 'apply_forensic_camera',
            mock.MagicMock(side_effect=AssertionError('forensic called')),
        ) as forensic_mock:
            response = await endpoint(
                request,
                image=_fake_upload(_jpeg_bytes()),
                config=None,
                output_format='jpeg',
                include_exif=False,
            )
        forensic_mock.assert_not_called()
        self.assertEqual(response.headers['X-Process'], 'noise')
        self.assertEqual(response.headers['X-Output-Format'], 'jpeg')
        self.assertTrue(response.body.startswith(b'\xff\xd8\xff'))

    async def test_perturb_stage_never_calls_forensic(self):
        endpoint = app_module._make_plain_endpoint('perturb')
        request = mock.MagicMock()
        request._form = None
        request.form = mock.AsyncMock(return_value={})
        with mock.patch.object(
            app_module, 'apply_forensic_camera',
            mock.MagicMock(side_effect=AssertionError('forensic called')),
        ) as forensic_mock:
            await endpoint(
                request,
                image=_fake_upload(_jpeg_bytes()),
                config=None,
                output_format='png',
                include_exif=False,
            )
        forensic_mock.assert_not_called()


class ForensicPackageSanityTests(unittest.TestCase):
    """One tiny real forensic call in degraded mode (no ExifTool needed)."""

    def test_apply_forensic_camera_writes_jpeg(self):
        tmpctx = tempfile.TemporaryDirectory(prefix='test_forensic_')
        with tmpctx as tmp:
            dest = os.path.join(tmp, 'out.png')  # wrong suffix on purpose
            with mock.patch(
                'image_postprocess.forensic_camera.pipeline.find_exiftool', return_value=None
            ):
                out_path = apply_forensic_camera(
                    Image.new('RGB', (8, 8), (10, 20, 30)), dest
                )
            self.assertTrue(str(out_path).endswith('.jpg'))
            self.assertTrue(os.path.isfile(out_path))
            with open(out_path, 'rb') as fh:
                head = fh.read(3)
            self.assertTrue(head.startswith(b'\xff\xd8\xff'))
            with Image.open(out_path) as im:
                self.assertEqual(im.format, 'JPEG')


if __name__ == '__main__':
    unittest.main()
