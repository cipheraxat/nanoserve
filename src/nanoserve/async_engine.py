from __future__ import annotations

import asyncio
import queue
import threading
import uuid
from collections.abc import AsyncIterator

from nanoserve.engine import LLMEngine, RequestOutput
from nanoserve.sampling import SamplingParams


class AsyncEngine:
    """Runs `LLMEngine.step()` in a background thread and fans results out to asyncio streams.

    The engine is only touched from its own thread. Request handlers talk to it through a
    thread-safe inbox, and each request gets an asyncio queue that the engine thread feeds
    with `call_soon_threadsafe`. A client that disconnects aborts its request, which frees
    its KV blocks on the next loop iteration.
    """

    def __init__(self, engine: LLMEngine):
        self.engine = engine
        self._inbox: queue.Queue = queue.Queue()
        self._streams: dict[str, tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = {}
        self._thread = threading.Thread(target=self._run, name="nanoserve-engine", daemon=True)
        self._stopped = threading.Event()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stopped.set()
        self._inbox.put(None)
        self._thread.join(timeout=5)

    def _handle(self, item) -> None:
        if item is None:
            return
        kind, request_id, *rest = item
        if kind == "add":
            token_ids, params = rest
            try:
                self.engine.add_request(token_ids, params, request_id)
            except ValueError as exc:
                self._deliver(request_id, exc)
        elif kind == "abort":
            self.engine.abort(request_id)

    def _deliver(self, request_id: str, item: RequestOutput | Exception) -> None:
        stream = self._streams.get(request_id)
        if stream is not None:
            loop, q = stream
            loop.call_soon_threadsafe(q.put_nowait, item)

    def _run(self) -> None:
        while not self._stopped.is_set():
            if not self.engine.has_unfinished():
                self._handle(self._inbox.get())  # sleep until there is work
            while True:
                try:
                    self._handle(self._inbox.get_nowait())
                except queue.Empty:
                    break
            if self._stopped.is_set():
                break
            try:
                outputs = self.engine.step()
            except Exception as exc:  # surface engine bugs to every waiting client
                for request_id in list(self._streams):
                    self._deliver(request_id, exc)
                    self.engine.abort(request_id)
                continue
            for out in outputs:
                self._deliver(out.request_id, out)

    async def generate(self, token_ids: list[int], params: SamplingParams) -> AsyncIterator[RequestOutput]:
        request_id = f"cmpl-{uuid.uuid4().hex[:24]}"
        q: asyncio.Queue = asyncio.Queue()
        self._streams[request_id] = (asyncio.get_running_loop(), q)
        self._inbox.put(("add", request_id, token_ids, params))
        finished = False
        try:
            while True:
                item = await q.get()
                if isinstance(item, Exception):
                    finished = True
                    raise item
                yield item
                if item.finished:
                    finished = True
                    return
        finally:
            self._streams.pop(request_id, None)
            if not finished:
                self._inbox.put(("abort", request_id))
