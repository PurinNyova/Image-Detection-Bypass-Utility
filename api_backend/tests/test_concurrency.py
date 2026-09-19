"""Concurrency tests: CPU work off the event loop, health responsive, transforms overlap."""

import asyncio
import io
import threading
import unittest
from unittest.mock import patch

import importlib

from PIL import Image
from starlette.datastructures import UploadFile

app_module = importlib.import_module('api_backend.app')

WAIT_INTERVAL = 0.01
SPIN_TIMEOUT = 5.0
RELEASE_TIMEOUT = 10.0


def _png_bytes(width=8, height=8):
    buf = io.BytesIO()
    Image.new('RGB', (width, height), 'red').save(buf, format='PNG')
    return buf.getvalue()


class _FakeRequest:
    async def form(self):
        return {}  # empty form -> _reject_extras sees no extra fields


class ConcurrencyTests(unittest.IsolatedAsyncioTestCase):

    def _make_entries(self, count=1):
        return [UploadFile(io.BytesIO(_png_bytes()), size=None) for _ in range(count)]

    async def _spin_until(self, predicate, what):
        deadline = asyncio.get_running_loop().time() + SPIN_TIMEOUT
        while not predicate():
            if asyncio.get_running_loop().time() > deadline:
                self.fail(f'timed out waiting for {what}')
            await asyncio.sleep(WAIT_INTERVAL)

    async def test_slow_transform_blocks_thread_health_responsive_and_overlap(self):
        loop_ids = {'loop': threading.get_ident()}
        started = []
        threads = []
        on_loop_violations = []
        release_a = threading.Event()
        release_b = threading.Event()

        def fake_a(arr, args, **kwargs):
            threads.append(('a', threading.get_ident()))
            if threading.get_ident() == loop_ids['loop']:
                on_loop_violations.append('a')
            started.append('a')
            # blocks only this worker thread; the event loop must stay free
            release_a.wait(RELEASE_TIMEOUT)
            return arr

        def fake_b(arr, args, **kwargs):
            threads.append(('b', threading.get_ident()))
            if threading.get_ident() == loop_ids['loop']:
                on_loop_violations.append('b')
            started.append('b')
            release_b.wait(RELEASE_TIMEOUT)
            return arr

        entry_a, entry_b = self._make_entries(2)
        request = _FakeRequest()

        async def run_stage(entry):
            return await app_module._handle_stage(
                stage_name='noise', request=request, image=entry, config_raw=None,
                output_format='png', include_exif=False,
                reference_image=None, lut_file=None,
                allowed_fields={'image', 'config', 'output_format', 'include_exif'},
            )

        try:
            with patch.object(app_module, 'process_array', fake_a):
                task_a = asyncio.create_task(run_stage(entry_a))
                try:
                    # slow "transform" has begun: its sync mock is running
                    await self._spin_until(lambda: started.count('a') == 1, 'mock A entry')

                    # while A is blocked in a worker thread, health must respond fast
                    health = await asyncio.wait_for(
                        asyncio.to_thread(app_module.health_check), timeout=1.0)
                    self.assertEqual(health, {'status': 'ok'})

                    # event loop also stays free: a trivial await completes fast
                    await asyncio.wait_for(self._noop(), timeout=1.0)

                    # second transform must enter its mock while A is still blocked -> overlap
                    with patch.object(app_module, 'process_array', fake_b):
                        task_b = asyncio.create_task(run_stage(entry_b))
                        try:
                            await self._spin_until(lambda: started.count('b') == 1, 'mock B entry')
                            self.assertIn('a', started)  # A still not finished (not released)
                            self.assertEqual(started.count('a'), 1)

                            ids = [t for _name, t in threads]
                            loop_id = loop_ids['loop']
                            self.assertEqual(len(set(ids)), len(ids), 'worker threads must be distinct')
                            for tid in ids:
                                self.assertNotEqual(tid, loop_id)
                                # acceptance: if process_array ran on the event loop,
                                # the loop would be blocked and health/spin would fail
                                # (mock would record the loop thread id here)
                            self.assertEqual(on_loop_violations, [])
                        finally:
                            release_b.set()
                    response_b = await asyncio.wait_for(task_b, timeout=RELEASE_TIMEOUT)

                    # normal task ordering; task_a above still awaited below
                    release_a.set()
                finally:
                    release_a.set()
                response_a = await asyncio.wait_for(task_a, timeout=RELEASE_TIMEOUT)
        finally:
            release_a.set()
            release_b.set()

        self.assertEqual(response_a.headers['X-Process'], 'noise')
        self.assertEqual(response_b.headers['X-Process'], 'noise')
        self.assertEqual(response_a.headers['X-Output-Format'], 'png')

    async def _noop(self):
        return None


if __name__ == '__main__':
    unittest.main()
