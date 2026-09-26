"""Exercise the ASGI boundary, including uploads without Content-Length."""
import asyncio
import unittest

from api.public_demo import PublicDemoGuard


class GuardTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.calls = 0
        self.received = None
        self.now = 100.0

        async def downstream(scope, receive, send):
            self.calls += 1
            self.received = await receive()
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"{}"})

        self.guard = PublicDemoGuard(downstream, max_body_bytes=10, clock=lambda: self.now)

    async def request(self, chunks=(b"ok",), headers=(), client="1.2.3.4", path="/api/demo/predict", receive=None):
        scope = {"type": "http", "method": "POST", "path": path,
                 "client": (client, 1234), "headers": list(headers)}
        messages = [{"type": "http.request", "body": chunk, "more_body": i < len(chunks) - 1}
                    for i, chunk in enumerate(chunks)]

        async def default_receive():
            return messages.pop(0) if messages else {"type": "http.disconnect"}

        output = []

        async def send(message):
            output.append(message)

        await self.guard(scope, receive or default_receive, send)
        return output

    async def test_chunked_limit_never_reaches_parser(self):
        output = await self.request(chunks=(b"12345", b"678901"))
        self.assertEqual(output[0]["status"], 413)
        self.assertEqual(self.calls, 0)
        self.assertEqual(self.guard.active, 0)

    async def test_valid_body_is_replayed_with_privacy_headers(self):
        output = await self.request(chunks=(b"12345", b"67890"))
        self.assertEqual(output[0]["status"], 200)
        self.assertEqual(self.received["body"], b"1234567890")
        self.assertEqual(dict(output[0]["headers"])[b"cache-control"], b"no-store")

    async def test_declared_oversize_rejected_without_reading(self):
        async def unread():
            self.fail("Oversized upload was read")
        output = await self.request(headers=[(b"content-length", b"11")], receive=unread)
        self.assertEqual(output[0]["status"], 413)

    async def test_invalid_or_false_content_lengths(self):
        for headers in [[(b"content-length", b"-1")], [(b"content-length", b"bad")],
                        [(b"content-length", b"2"), (b"content-length", b"2")],
                        [(b"content-length", b"1")]]:
            output = await self.request(headers=headers)
            self.assertEqual(output[0]["status"], 400)
        output = await self.request(chunks=(b"12345678901",), headers=[(b"content-length", b"1")])
        self.assertEqual(output[0]["status"], 413)
        self.assertEqual(self.calls, 0)

    async def test_rate_limit_expiry_and_forged_forwarding_headers(self):
        self.guard.requests_per_minute = 2
        await self.request()
        await self.request()
        output = await self.request(headers=[(b"x-forwarded-for", b"5.6.7.8")])
        self.assertEqual(output[0]["status"], 429)
        self.assertEqual(dict(output[0]["headers"])["Retry-After".lower().encode()], b"60")
        self.assertEqual((await self.request(client="5.6.7.8"))[0]["status"], 200)
        self.now += 60
        self.assertEqual((await self.request())[0]["status"], 200)

    async def test_client_state_is_bounded_and_expires(self):
        self.guard.max_clients = 1
        await self.request()
        self.assertEqual((await self.request(client="other"))[0]["status"], 503)
        self.now += 61
        self.assertEqual((await self.request(client="other"))[0]["status"], 200)
        self.assertEqual(len(self.guard.clients), 1)

    async def test_concurrent_upload_admission(self):
        self.guard.max_active = 1
        entered, release = asyncio.Event(), asyncio.Event()

        async def slow_receive():
            entered.set()
            await release.wait()
            return {"type": "http.request", "body": b"ok", "more_body": False}

        first = asyncio.create_task(self.request(receive=slow_receive))
        await entered.wait()
        try:
            self.assertEqual((await self.request())[0]["status"], 503)
        finally:
            release.set()
            await first
        self.assertEqual(self.guard.active, 0)
        self.assertEqual((await self.request())[0]["status"], 200)

    async def test_slow_upload_and_disconnect_release_slot(self):
        self.guard.upload_timeout = .01

        async def stalled():
            await asyncio.Event().wait()

        self.assertEqual((await self.request(receive=stalled))[0]["status"], 408)

        async def disconnected():
            return {"type": "http.disconnect"}

        self.assertEqual(await self.request(receive=disconnected), [])
        self.assertEqual(self.guard.active, 0)
        self.assertEqual(self.calls, 0)

    async def test_redirect_variant_is_also_guarded(self):
        output = await self.request(chunks=(b"12345678901",), path="/api/demo/predict/")
        self.assertEqual(output[0]["status"], 413)


if __name__ == "__main__":
    unittest.main()
