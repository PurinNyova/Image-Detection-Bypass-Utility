import unittest
from contextlib import ExitStack
from unittest import mock

from fastapi import HTTPException

from api_backend.app import (
    MAX_UPLOAD_BYTES,
    _UPLOAD_CHUNK,
    _call_stage,
    _read_bounded,
)


class FakeUpload:
    def __init__(self, data: bytes, filename: str | None = None):
        self._data = data
        self.read_calls = 0
        self.filename = filename

    async def read(self, size: int = -1) -> bytes:
        self.read_calls += 1
        if not self._data:
            return b''
        chunk, self._data = self._data[:size], self._data[size:]
        return chunk

    def assert_not_read(self):
        assert self.read_calls == 0, f'read awaited {self.read_calls} times'


class _FakeLenChunk(bytes):
    """1-byte buffer that reports an arbitrary length to `len()`."""

    def __new__(cls, fake_len: int):
        obj = super().__new__(cls, b'\x00')
        obj._fake_len = fake_len
        return obj

    def __len__(self) -> int:
        return self._fake_len


class TestUploadCap(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Patch the cap small so boundary bytes are real, tiny, and fast.
        self._patcher = mock.patch('api_backend.app.MAX_UPLOAD_BYTES', 12)
        self._patcher.start()
        self.addCleanup(self._patcher.stop)

    async def test_constant_is_64_mib(self):
        self.assertEqual(MAX_UPLOAD_BYTES, 64 * 1024 * 1024)
        self.assertEqual(_UPLOAD_CHUNK, 1024 * 1024)

    async def test_boundary_bytes_accepted(self):
        upload = FakeUpload(b'A' * 12)
        self.assertEqual(await _read_bounded(upload, 'image'), b'A' * 12)

    async def test_one_byte_over_rejected_413(self):
        upload = FakeUpload(b'A' * 13)
        with self.assertRaises(HTTPException) as ctx:
            await _read_bounded(upload, 'image')
        self.assertEqual(ctx.exception.status_code, 413)

    async def test_empty_400(self):
        with self.assertRaises(HTTPException) as ctx:
            await _read_bounded(FakeUpload(b''), 'image')
        self.assertEqual(ctx.exception.status_code, 400)

    async def test_none_400(self):
        with self.assertRaises(HTTPException) as ctx:
            await _read_bounded(None, 'image')
        self.assertEqual(ctx.exception.status_code, 400)

    async def test_reads_in_chunks(self):
        data = b'X' * 8
        upload = FakeUpload(data)
        await _read_bounded(upload, 'image')
        self.assertGreaterEqual(upload.read_calls, 2)


class TestOversizedNoBigAllocation(unittest.IsolatedAsyncioTestCase):
    """Oversized rejection via fake chunks reporting >cap length; nothing >cap is allocated."""

    def test_chunk_subclass_reports_fake_len(self):
        self.assertEqual(len(_FakeLenChunk(MAX_UPLOAD_BYTES + 1)), MAX_UPLOAD_BYTES + 1)

    async def test_oversize_primary_raises_413(self):
        upload = mock.AsyncMock()
        upload.read = mock.AsyncMock(return_value=_FakeLenChunk(MAX_UPLOAD_BYTES + 1))
        with self.assertRaises(HTTPException) as ctx:
            await _read_bounded(upload, 'image')
        self.assertEqual(ctx.exception.status_code, 413)
        self.assertEqual(ctx.exception.detail, "image exceeds 67108864 byte upload cap")

    async def test_oversize_reference_raises_413(self):
        upload = mock.AsyncMock()
        upload.read = mock.AsyncMock(return_value=_FakeLenChunk(MAX_UPLOAD_BYTES + 1))
        with self.assertRaises(HTTPException) as ctx:
            await _read_bounded(upload, 'reference_image')
        self.assertEqual(ctx.exception.status_code, 413)
        self.assertIn('reference_image', ctx.exception.detail)

    async def test_oversize_lut_raises_413(self):
        upload = mock.AsyncMock()
        upload.read = mock.AsyncMock(return_value=_FakeLenChunk(MAX_UPLOAD_BYTES + 1))
        with self.assertRaises(HTTPException) as ctx:
            await _read_bounded(upload, 'lut_file')
        self.assertEqual(ctx.exception.status_code, 413)
        self.assertIn('lut_file', ctx.exception.detail)

    async def test_second_chunk_pushes_over_cap(self):
        upload = mock.AsyncMock()
        upload.read = mock.AsyncMock(
            side_effect=[_FakeLenChunk(MAX_UPLOAD_BYTES), b'z'])
        with self.assertRaises(HTTPException) as ctx:
            await _read_bounded(upload, 'image')
        self.assertEqual(ctx.exception.status_code, 413)


class TestUnusedSecondaryNotDecoded(unittest.IsolatedAsyncioTestCase):
    def _request(self):
        request = mock.MagicMock()
        request.form = mock.AsyncMock(return_value={})
        return request

    def start_mocks(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.decode = stack.enter_context(
            mock.patch('api_backend.app._decode_rgb', return_value=mock.MagicMock()))
        self.load_lut = stack.enter_context(mock.patch('api_backend.app.load_lut_bytes'))
        stack.enter_context(mock.patch('api_backend.app.build_processing_args'))
        stack.enter_context(mock.patch('api_backend.app.process_array',
                                       return_value=mock.MagicMock()))
        stack.enter_context(mock.patch('api_backend.app.encode_image_array',
                                       return_value=(b'out', 'image/png')))

    def _extras(self):
        # Real bounded uploads; "unused" checks assert .read was never awaited.
        return FakeUpload(b'ref'), FakeUpload(b'lut', filename='upload.lut')

    async def test_noise_ignores_reference_and_lut(self):
        self.start_mocks()
        reference, lut = self._extras()
        await _call_stage('noise', self._request(), FakeUpload(b'img'), '{}', 'png', False,
                          reference, lut)
        reference.assert_not_read()
        self.assertEqual(self.decode.call_count, 1)  # primary only; ref/lut never decoded
        self.load_lut.assert_not_called()            # lut bounded-read but never loaded

    async def test_fft_reads_reference_never_lut(self):
        self.start_mocks()
        reference, lut = self._extras()
        await _call_stage('fft', self._request(), FakeUpload(b'img'), '{}', 'png', False,
                          reference, lut)
        self.assertGreater(reference.read_calls, 0)
        self.load_lut.assert_not_called()  # lut bounded-read but never loaded/decoded
        self.assertEqual(self.decode.call_count, 2)  # reference + primary

    async def test_lut_reads_lut_never_reference(self):
        self.start_mocks()
        reference, lut = self._extras()
        await _call_stage('lut', self._request(), FakeUpload(b'img'), '{}', 'png', False,
                          reference, lut)
        reference.assert_not_read()
        self.assertGreater(lut.read_calls, 0)
        self.load_lut.assert_called_once()
        self.assertEqual(self.decode.call_count, 1)  # reference ignored, primary decoded


if __name__ == '__main__':
    unittest.main()
