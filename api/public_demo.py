"""Bound public uploads before multipart parsing or model execution.

Limits are per process. Use one worker, or add shared limits at the ingress.
Client addresses come from ASGI scope, never directly from forwarding headers.
"""

import asyncio
import math
import time
from collections import OrderedDict, deque

from starlette.responses import JSONResponse


class PublicDemoGuard:
    def __init__(self, app, *, max_body_bytes=10 * 1024 * 1024 + 64 * 1024,
                 requests_per_minute=10, max_active=2, upload_timeout=20,
                 max_clients=10000, clock=time.monotonic):
        self.app = app
        self.max_body_bytes = max_body_bytes
        self.requests_per_minute = requests_per_minute
        self.max_active = max_active
        self.upload_timeout = upload_timeout
        self.max_clients = max_clients
        self.clock = clock
        self.clients = OrderedDict()
        self.active = 0

    async def __call__(self, scope, receive, send):
        if (scope["type"] != "http" or scope["method"] != "POST"
                or scope["path"].rstrip("/") != "/api/demo/predict"):
            return await self.app(scope, receive, send)

        async def reply(status, detail, retry=None):
            headers = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}
            if retry is not None:
                headers["Retry-After"] = str(retry)
            await JSONResponse({"detail": detail}, status_code=status, headers=headers)(scope, receive, send)

        now = self.clock()
        # Ordered by most recent attempt, so idle client state expires in bounded work.
        while self.clients and next(iter(self.clients.values()))[-1] <= now - 60:
            self.clients.popitem(last=False)
        client = (scope.get("client") or ("unknown", 0))[0]
        attempts = self.clients.get(client)
        if attempts is None:
            if len(self.clients) >= self.max_clients:
                return await reply(503, "PetSnap is busy. Please try again shortly.", 60)
            attempts = self.clients[client] = deque()
        while attempts and attempts[0] <= now - 60:
            attempts.popleft()
        if len(attempts) >= self.requests_per_minute:
            return await reply(429, "You’ve met a lot of dogs! Please wait before trying again.",
                               max(1, math.ceil(60 - (now - attempts[0]))))
        attempts.append(now)
        self.clients.move_to_end(client)

        lengths = [value for name, value in scope["headers"] if name.lower() == b"content-length"]
        if lengths:
            if len(lengths) != 1 or not lengths[0].isdigit():
                return await reply(400, "Invalid request length.")
            if len(lengths[0]) > 12 or int(lengths[0]) > self.max_body_bytes:
                return await reply(413, "Choose an image smaller than 10 MB.")
        if self.active >= self.max_active:
            return await reply(503, "PetSnap is meeting another dog. Try again in a moment.", 2)

        self.active += 1
        try:
            # Count actual bytes even if Content-Length is absent or incorrect.
            # Nothing reaches the multipart parser until the entire bounded body arrives.
            body = bytearray()

            async def read_body():
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        return "disconnected"
                    chunk = message.get("body", b"")
                    if len(body) + len(chunk) > self.max_body_bytes:
                        return "oversize"
                    body.extend(chunk)
                    if not message.get("more_body", False):
                        return "complete"

            try:
                outcome = await asyncio.wait_for(read_body(), self.upload_timeout)
            except asyncio.TimeoutError:
                return await reply(408, "The upload took too long. Please try again.")
            if outcome == "disconnected":
                return
            if outcome == "oversize":
                return await reply(413, "Choose an image smaller than 10 MB.")
            if lengths and int(lengths[0]) != len(body):
                return await reply(400, "Request length does not match the upload.")

            delivered = False

            async def replay():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    payload = bytes(body)
                    body.clear()
                    return {"type": "http.request", "body": payload, "more_body": False}
                return await receive()

            async def private_response(message):
                if message["type"] == "http.response.start":
                    message = dict(message)
                    headers = [(k, v) for k, v in message.get("headers", [])
                               if k.lower() not in (b"cache-control", b"x-content-type-options")]
                    message["headers"] = headers + [(b"cache-control", b"no-store"),
                                                     (b"x-content-type-options", b"nosniff")]
                await send(message)

            await self.app(scope, replay, private_response)
        finally:
            self.active -= 1
