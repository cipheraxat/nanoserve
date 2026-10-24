"""OpenAI-compatible HTTP server: /v1/completions, /v1/chat/completions (with SSE streaming),
/v1/models, /health and Prometheus /metrics."""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from nanoserve.async_engine import AsyncEngine
from nanoserve.config import EngineConfig
from nanoserve.engine import LLMEngine, RequestOutput
from nanoserve.sampling import SamplingParams


class _SamplingFields(BaseModel):
    model: str | None = None
    max_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    seed: int | None = None
    stream: bool = False
    ignore_eos: bool = False

    def sampling_params(self) -> SamplingParams:
        return SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            seed=self.seed,
            ignore_eos=self.ignore_eos,
        )


class CompletionRequest(_SamplingFields):
    prompt: str | list[int]


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(_SamplingFields):
    messages: list[ChatMessage] = Field(min_length=1)


class IncrementalDetokenizer:
    """Turns a growing list of token ids into text deltas, holding back a trailing partial
    UTF-8 character until the token that completes it arrives."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.text = ""

    def delta(self, token_ids: list[int], final: bool) -> str:
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        if text.endswith("�") and not final:
            return ""
        delta, self.text = text[len(self.text) :], text
        return delta


class ServerMetrics:
    def __init__(self) -> None:
        self.requests_total = 0
        self.ttft_sum = 0.0
        self.ttft_count = 0
        self.e2e_sum = 0.0
        self.e2e_count = 0


def create_app(engine: LLMEngine) -> FastAPI:
    async_engine = AsyncEngine(engine)
    metrics = ServerMetrics()
    tokenizer = engine.tokenizer
    model_name = engine.config.model

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async_engine.start()
        yield
        async_engine.stop()

    app = FastAPI(title="nanoserve", lifespan=lifespan)

    def prepare(token_ids: list[int], params: SamplingParams) -> None:
        try:
            engine.validate_request(token_ids, params)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    async def run(token_ids: list[int], params: SamplingParams) -> AsyncIterator[tuple[RequestOutput, str]]:
        """Yield (output, text delta) per generated token and record latency metrics."""
        metrics.requests_total += 1
        start = time.perf_counter()
        first = True
        detok = IncrementalDetokenizer(tokenizer)
        async for out in async_engine.generate(token_ids, params):
            if first:
                metrics.ttft_sum += time.perf_counter() - start
                metrics.ttft_count += 1
                first = False
            yield out, detok.delta(out.output_token_ids, out.finished)
        metrics.e2e_sum += time.perf_counter() - start
        metrics.e2e_count += 1

    def usage(out: RequestOutput) -> dict:
        return {
            "prompt_tokens": out.num_prompt_tokens,
            "completion_tokens": len(out.output_token_ids),
            "total_tokens": out.num_prompt_tokens + len(out.output_token_ids),
            "prompt_tokens_details": {"cached_tokens": out.num_cached_tokens},
        }

    def sse(payload: dict) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    def parse_params(req: _SamplingFields) -> SamplingParams:
        try:
            return req.sampling_params()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        params = parse_params(req)
        token_ids = tokenizer.encode(req.prompt) if isinstance(req.prompt, str) else req.prompt
        prepare(token_ids, params)
        created = int(time.time())

        if req.stream:

            async def stream():
                async for out, delta in run(token_ids, params):
                    choice = {
                        "index": 0,
                        "text": delta,
                        "finish_reason": out.finish_reason.value if out.finished else None,
                    }
                    chunk = {
                        "id": out.request_id,
                        "object": "text_completion",
                        "created": created,
                        "model": model_name,
                        "choices": [choice],
                    }
                    if out.finished:
                        chunk["usage"] = usage(out)
                    yield sse(chunk)
                yield "data: [DONE]\n\n"

            return StreamingResponse(stream(), media_type="text/event-stream")

        text, last = "", None
        async for out, delta in run(token_ids, params):
            text, last = text + delta, out
        return {
            "id": last.request_id,
            "object": "text_completion",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "text": text, "finish_reason": last.finish_reason.value}],
            "usage": usage(last),
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        params = parse_params(req)
        messages = [m.model_dump() for m in req.messages]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        prepare(token_ids, params)
        created = int(time.time())

        if req.stream:

            async def stream():
                first = True
                async for out, delta in run(token_ids, params):
                    message = {"role": "assistant", "content": delta} if first else {"content": delta}
                    first = False
                    choice = {
                        "index": 0,
                        "delta": message,
                        "finish_reason": out.finish_reason.value if out.finished else None,
                    }
                    chunk = {
                        "id": out.request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [choice],
                    }
                    if out.finished:
                        chunk["usage"] = usage(out)
                    yield sse(chunk)
                yield "data: [DONE]\n\n"

            return StreamingResponse(stream(), media_type="text/event-stream")

        text, last = "", None
        async for out, delta in run(token_ids, params):
            text, last = text + delta, out
        return {
            "id": last.request_id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": last.finish_reason.value,
                }
            ],
            "usage": usage(last),
        }

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "nanoserve"}]}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/metrics")
    async def prometheus_metrics():
        bm, sched, stats = engine.block_manager, engine.scheduler, engine.stats
        lines = [
            ("gauge", "nanoserve_num_requests_running", len(sched.running)),
            ("gauge", "nanoserve_num_requests_waiting", len(sched.waiting)),
            ("gauge", "nanoserve_kv_cache_usage_ratio", round(bm.usage, 4)),
            ("gauge", "nanoserve_prefix_cache_hit_rate", round(bm.prefix_hit_rate, 4)),
            ("counter", "nanoserve_prompt_tokens_total", stats.prompt_tokens),
            ("counter", "nanoserve_generation_tokens_total", stats.generated_tokens),
            ("counter", "nanoserve_prefix_cache_hit_tokens_total", bm.prefix_hit_tokens),
            ("counter", "nanoserve_preemptions_total", sched.num_preemptions),
            ("counter", "nanoserve_requests_total", metrics.requests_total),
            ("counter", "nanoserve_engine_steps_total", stats.steps),
        ]
        body = []
        for kind, name, value in lines:
            body += [f"# TYPE {name} {kind}", f"{name} {value}"]
        for name, total, count in (
            ("nanoserve_time_to_first_token_seconds", metrics.ttft_sum, metrics.ttft_count),
            ("nanoserve_e2e_request_latency_seconds", metrics.e2e_sum, metrics.e2e_count),
        ):
            body += [f"# TYPE {name} summary", f"{name}_sum {total:.6f}", f"{name}_count {count}"]
        return PlainTextResponse("\n".join(body) + "\n", media_type="text/plain; version=0.0.4")

    @app.exception_handler(ValueError)
    async def value_error(_: Request, exc: ValueError):
        return JSONResponse(status_code=400, content={"error": {"message": str(exc), "type": "invalid_request_error"}})

    return app


def parse_args(argv: list[str] | None = None) -> tuple[EngineConfig, argparse.Namespace]:
    parser = argparse.ArgumentParser(description="Serve a Qwen3 model with an OpenAI-compatible API")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--kv-cache-memory-gb", type=float, default=1.0)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--no-prefix-caching", action="store_true")
    parser.add_argument("--attention-backend", default="auto", choices=["auto", "torch", "triton"])
    args = parser.parse_args(argv)
    config = EngineConfig(
        model=args.model,
        device=args.device,
        dtype=args.dtype,
        block_size=args.block_size,
        kv_cache_memory_gb=args.kv_cache_memory_gb,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        enable_prefix_caching=not args.no_prefix_caching,
        attention_backend=args.attention_backend,
    )
    return config, args


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config, args = parse_args(argv)
    app = create_app(LLMEngine(config))
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
