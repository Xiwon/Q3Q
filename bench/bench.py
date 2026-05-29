"""
Benchmark harness for Q3Q implementations

Measures three things for each combo:
    1. prefill latency for the given prompt length
    2. steady-state decode throughput (tok/s)
    3. peak GPU memory across the whole prefill+decode cycle

Usage:
    python -m bench.bench --impl eager_split --model_path PATH \
        --prompt_lens 16,128,512 --decode_steps 32 --warmup 2 --repeat 3

Each repeat fully resets the model state. Results are printed as a table and
optionally appended as JSONL to `--out`.
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

IMPLS = {
    "eager_main": "eager_main",
    "eager_split": "eager_split",
}

def _load_impl(name: str):
    if name not in IMPLS:
        raise ValueError(f"unknown impl `{name}`. choices: {list(IMPLS)}")
    return importlib.import_module(IMPLS[name])

def _build_model(impl_module, model_path: str, device: str):
    torch.set_default_dtype(torch.bfloat16)
    config = impl_module.QMoeConfig(str(Path(model_path) / "config.json"))
    with torch.device("meta"):
        model = impl_module.QMoeEngine(config)
    torch.set_default_dtype(torch.float32)
    model.to_empty(device=device)
    impl_module.load_safetensors_weights(model, model_path, device=device)
    model.eval()
    return model, config

def _make_prompt_ids(vocab_size: int, length: int, device: str) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(0)
    ids = torch.randint(low=10, high=vocab_size, size=(1, length), generator=g)
    return ids.to(device)

def _cuda_event_pair():
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

@torch.no_grad()
def _run_once(model, prompt_ids: torch.Tensor, decode_steps: int):
    """Returns (prefill_ms, decode_ms_total, decode_steps_done, peak_bytes)."""
    has_split = hasattr(model, "prefill") and hasattr(model, "decode_step")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    pe_start, pe_end = _cuda_event_pair()
    pe_start.record()
    if has_split:
        logits, state = model.prefill(prompt_ids)
    else:
        logits = model(prompt_ids)
        state = {"running": prompt_ids}
    pe_end.record()
    torch.cuda.synchronize()
    prefill_ms = pe_start.elapsed_time(pe_end)

    cur_logits = logits[:, -1, :]
    next_tok = cur_logits.float().argmax(dim=-1, keepdim=True)

    de_start, de_end = _cuda_event_pair()
    de_start.record()
    steps_done = 0
    for _ in range(decode_steps):
        if has_split:
            logits, state = model.decode_step(next_tok, state)
            cur_logits = logits[:, -1, :]
        else:
            state["running"] = torch.cat([state["running"], next_tok], dim=-1)
            logits = model(state["running"])
            cur_logits = logits[:, -1, :]
        next_tok = cur_logits.float().argmax(dim=-1, keepdim=True)
        steps_done += 1
    de_end.record()
    torch.cuda.synchronize()
    decode_ms_total = de_start.elapsed_time(de_end)

    peak_bytes = torch.cuda.max_memory_allocated()
    return prefill_ms, decode_ms_total, steps_done, peak_bytes

def benchmark(args):
    device = args.device
    impl = _load_impl(args.impl)
    model, config = _build_model(impl, args.model_path, device)
    vocab_size = config.vocab_size

    prompt_lens = [int(x) for x in args.prompt_lens.split(",") if x.strip()]
    rows = []

    for L in prompt_lens:
        prompt_ids = _make_prompt_ids(vocab_size, L, device)

        for _ in range(args.warmup):
            _run_once(model, prompt_ids, max(1, args.decode_steps // 4))

        prefill_ms_list = []
        decode_tok_per_s_list = []
        peak_gib_list = []
        for _ in range(args.repeat):
            pf_ms, de_ms, steps_done, peak = _run_once(model, prompt_ids, args.decode_steps)
            prefill_ms_list.append(pf_ms)
            tok_per_s = steps_done * 1000.0 / de_ms if de_ms > 0 else float("nan")
            decode_tok_per_s_list.append(tok_per_s)
            peak_gib_list.append(peak / (1024**3))

        def _stats(xs):
            xs_s = sorted(xs)
            return {
                "min": xs_s[0],
                "median": xs_s[len(xs_s) // 2],
                "max": xs_s[-1],
            }

        row = {
            "impl": args.impl,
            "prompt_len": L,
            "decode_steps": args.decode_steps,
            "prefill_ms": _stats(prefill_ms_list),
            "decode_tok_per_s": _stats(decode_tok_per_s_list),
            "peak_gib": _stats(peak_gib_list),
        }
        rows.append(row)
        print(
            f"[{args.impl}] T_in={L:<5d} "
            f"prefill={row['prefill_ms']['median']:7.1f} ms  "
            f"decode={row['decode_tok_per_s']['median']:6.2f} tok/s  "
            f"peak={row['peak_gib']['median']:5.2f} GiB"
        )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "a") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"[bench] appended {len(rows)} rows -> {out}")

def main():
    p = argparse.ArgumentParser(description="Q3Q benchmark harness")
    p.add_argument("--impl", type=str, default="eager_split", choices=list(IMPLS))
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--prompt_lens", type=str, default="16,128,512")
    p.add_argument("--decode_steps", type=int, default=32)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--out", type=str, default=None, help="optional JSONL path to append results")
    args = p.parse_args()
    if not args.device.startswith("cuda"):
        raise RuntimeError("benchmark requires cuda device")
    benchmark(args)

if __name__ == "__main__":
    main()
