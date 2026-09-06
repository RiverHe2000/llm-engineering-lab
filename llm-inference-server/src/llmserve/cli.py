"""``llmserve {serve,generate,bench}``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from llmserve.benchmark import environment_info, render_markdown, run_benchmark
from llmserve.config import Settings
from llmserve.engine import GenerationEngine, GenerationRequest
from llmserve.logging_utils import configure_logging
from llmserve.sampling import SamplingParams


def _settings_from_args(args: argparse.Namespace) -> Settings:
    overrides: dict[str, Any] = {}
    for key in ("model_name", "device", "max_batch_size", "batch_window_ms", "max_context"):
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value
    if getattr(args, "quantize_int8", False):
        overrides["quantize_int8"] = True
    if getattr(args, "torch_compile", False):
        overrides["torch_compile"] = True
    return Settings(**overrides)


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from llmserve.api import create_app

    settings = _settings_from_args(args)
    configure_logging(settings.log_level, json_lines=not args.plain_logs)
    uvicorn.run(create_app(settings), host=args.host, port=args.port, log_config=None)
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    settings = _settings_from_args(args)
    configure_logging(settings.log_level, json_lines=False)
    engine = GenerationEngine.from_settings(settings)
    requests = [
        GenerationRequest(
            request_id=f"cli-{i}",
            prompt=p,
            max_new_tokens=args.max_tokens,
            sampling=SamplingParams(
                temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, seed=args.seed
            ),
        )
        for i, p in enumerate(args.prompt)
    ]
    for r in engine.generate_batch(requests):
        sys.stdout.write(
            f"[{r.request_id}] finish={r.finish_reason} tokens={r.completion_tokens} "
            f"latency={r.latency_ms:.0f}ms\n{r.text}\n\n"
        )
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    settings = _settings_from_args(args)
    configure_logging(settings.log_level, json_lines=False)
    engine = GenerationEngine.from_settings(settings)
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    rows = run_benchmark(
        engine,
        prompt=args.prompt,
        batch_sizes=batch_sizes,
        max_new_tokens=args.max_tokens,
        repeats=args.repeats,
    )
    env = environment_info(engine)
    if settings.quantize_int8:
        env["quantization"] = "dynamic int8"
    md = render_markdown(rows, args.title or f"{settings.model_name} on {env['device']}", env)
    sys.stdout.write(md)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if args.append else "w"
        with out.open(mode, encoding="utf-8") as fh:
            fh.write(md + "\n")
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps({"env": env, "rows": [r.to_dict() for r in rows]}, indent=2),
            encoding="utf-8",
        )
    return 0


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model-name", dest="model_name", default=None)
    p.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    p.add_argument("--max-context", dest="max_context", type=int, default=None)
    p.add_argument("--quantize-int8", dest="quantize_int8", action="store_true")
    p.add_argument("--torch-compile", dest="torch_compile", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="llmserve", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("serve", help="run the HTTP server")
    _add_model_args(s)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--max-batch-size", dest="max_batch_size", type=int, default=None)
    s.add_argument("--batch-window-ms", dest="batch_window_ms", type=float, default=None)
    s.add_argument("--plain-logs", action="store_true", help="human-readable instead of JSON")
    s.set_defaults(func=cmd_serve)

    g = sub.add_parser("generate", help="generate completions from the command line")
    _add_model_args(g)
    g.add_argument("--prompt", action="append", required=True)
    g.add_argument("--max-tokens", type=int, default=64)
    g.add_argument("--temperature", type=float, default=0.7)
    g.add_argument("--top-k", type=int, default=50)
    g.add_argument("--top-p", type=float, default=0.95)
    g.add_argument("--seed", type=int, default=None)
    g.set_defaults(func=cmd_generate)

    b = sub.add_parser("bench", help="throughput/latency across batch sizes")
    _add_model_args(b)
    b.add_argument("--prompt", default="The Reserve Bank of Australia said on Tuesday that")
    b.add_argument("--batch-sizes", default="1,2,4,8,16")
    b.add_argument("--max-tokens", type=int, default=64)
    b.add_argument("--repeats", type=int, default=3)
    b.add_argument("--title", default=None)
    b.add_argument("--out", default=None, help="append/write Markdown here")
    b.add_argument("--append", action="store_true")
    b.add_argument("--json-out", default=None)
    b.set_defaults(func=cmd_bench)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
