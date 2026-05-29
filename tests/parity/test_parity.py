"""
Parity harness for Q3Q implementations

Usage:
    # dump reference from the eager baseline
    python -m tests.parity.test_parity save \
        --impl eager_main --model_path PATH --out tests/parity/ref_eager.pt

    # check another impl against that reference
    python -m tests.parity.test_parity check \
        --impl eager_split --model_path PATH --ref tests/parity/ref_eager.pt

The harness drives prefill + greedy decoding directly,
records every step's next-token logits and the resulting token id, and compares
two runs at both the logits and the token-id level.
"""

import argparse
import importlib
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

IMPLS = { # registry of implementation package here
    "eager_main": "eager_main",
    "eager_split": "eager_split",
}

DEFAULT_PROMPT = (
    "From now on, you are a cute, energetic anime catgirl. "
    "You love cuddling, playing with yarn, and eating fish. "
    "Add 'Nya~' to the end of your sentences occasionally. "
    "Be highly affectionate, loyal to me, and treat me like your favorite human. "
    "If you understand, please reply a single sentence with an appropriate "
    "kaomojis (Japanese emoticons) to describe the feeling from your responses."
)
DEFAULT_NEW_TOKENS = 64

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

def _tokenize(model_path: str, prompt: str, device: str) -> torch.Tensor:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    return tok(prompt, return_tensors="pt").input_ids.to(device)

@torch.no_grad()
def run_greedy(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
):
    """drive the model with prefill + greedy argmax. Returns dict with:
        prompt_ids:    (1, T_in)
        token_ids:     (max_new_tokens,)  — generated token ids in order
        step_logits:   (max_new_tokens, V) fp32 — next-token logits at each step
    """
    device = input_ids.device
    V = model.lm_head.weight.shape[0]

    has_split = hasattr(model, "prefill") and hasattr(model, "decode_step")

    token_ids = []
    step_logits = torch.empty((max_new_tokens, V), dtype=torch.float32, device=device)

    if has_split:
        logits, state = model.prefill(input_ids)
        cur_logits = logits[:, -1, :]
    else:
        logits = model(input_ids)
        cur_logits = logits[:, -1, :]
        running = input_ids

    for step in range(max_new_tokens):
        step_logits[step] = cur_logits.float().squeeze(0)
        next_tok = cur_logits.float().argmax(dim=-1, keepdim=True)
        token_ids.append(int(next_tok.item()))
        if step == max_new_tokens - 1:
            break
        if has_split:
            logits, state = model.decode_step(next_tok, state)
            cur_logits = logits[:, -1, :]
        else:
            running = torch.cat([running, next_tok], dim=-1)
            logits = model(running)
            cur_logits = logits[:, -1, :]

    return {
        "prompt_ids": input_ids.detach().cpu(),
        "token_ids": torch.tensor(token_ids, dtype=torch.long),
        "step_logits": step_logits.detach().cpu(),
    }

def cmd_save(args):
    device = args.device
    impl = _load_impl(args.impl)
    model, _ = _build_model(impl, args.model_path, device)
    input_ids = _tokenize(args.model_path, args.prompt, device)

    out = run_greedy(model, input_ids, args.max_new_tokens)
    out["meta"] = {
        "impl": args.impl,
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "model_path": str(Path(args.model_path).resolve()),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(f"[save] wrote reference -> {out_path}")
    print(f"[save] generated token ids: {out['token_ids'].tolist()}")

def cmd_check(args):
    device = args.device
    ref = torch.load(args.ref, map_location="cpu", weights_only=False)
    meta = ref.get("meta", {})
    prompt = args.prompt or meta.get("prompt", DEFAULT_PROMPT)
    max_new_tokens = args.max_new_tokens or meta.get("max_new_tokens", DEFAULT_NEW_TOKENS)
    print(f"[check] reference impl={meta.get('impl')} prompt={prompt!r} new_tokens={max_new_tokens}")

    impl = _load_impl(args.impl)
    model, _ = _build_model(impl, args.model_path, device)
    input_ids = _tokenize(args.model_path, prompt, device)
    if not torch.equal(input_ids.cpu(), ref["prompt_ids"]):
        raise RuntimeError(
            f"prompt token id mismatch (tokenizer drift?). "
            f"got {input_ids.cpu().tolist()} vs ref {ref['prompt_ids'].tolist()}"
        )

    out = run_greedy(model, input_ids, max_new_tokens)

    ref_ids = ref["token_ids"]
    new_ids = out["token_ids"]
    ref_logits = ref["step_logits"]
    new_logits = out["step_logits"]

    ids_match = torch.equal(ref_ids, new_ids)
    if ids_match:
        first_div = -1
    else:
        diff = (ref_ids != new_ids).nonzero(as_tuple=False)
        first_div = int(diff[0].item()) if diff.numel() > 0 else -1

    abs_diff = (ref_logits - new_logits).abs()
    rel_diff = abs_diff / (ref_logits.abs() + 1e-6)
    max_abs = float(abs_diff.max())
    max_rel = float(rel_diff.max())
    mean_abs = float(abs_diff.mean())

    print(f"[check] token-id match:   {ids_match}")
    if not ids_match:
        print(f"        first divergent step: {first_div}")
        print(f"        ref ids: {ref_ids.tolist()}")
        print(f"        new ids: {new_ids.tolist()}")
    print(f"[check] logits max |Δ|:   {max_abs:.4e}")
    print(f"        logits mean |Δ|:  {mean_abs:.4e}")
    print(f"        logits max relΔ:  {max_rel:.4e}")

    ok = ids_match and max_abs <= args.atol
    print(f"[check] result: {'PASS' if ok else 'FAIL'} (atol={args.atol})")
    sys.exit(0 if ok else 1)

def build_parser():
    p = argparse.ArgumentParser(description="Q3Q parity harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model_path", type=str, required=True)
    common.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    s_save = sub.add_parser("save", parents=[common], help="dump reference from one impl")
    s_save.add_argument("--impl", type=str, default="eager_main", choices=list(IMPLS))
    s_save.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    s_save.add_argument("--max_new_tokens", type=int, default=DEFAULT_NEW_TOKENS)
    s_save.add_argument("--out", type=str, default="tests/parity/ref_eager.pt")
    s_save.set_defaults(func=cmd_save)

    s_check = sub.add_parser("check", parents=[common], help="compare an impl against a reference")
    s_check.add_argument("--impl", type=str, required=True, choices=list(IMPLS))
    s_check.add_argument("--ref", type=str, required=True)
    s_check.add_argument("--prompt", type=str, default=None, help="overrides ref meta if given")
    s_check.add_argument("--max_new_tokens", type=int, default=None)
    s_check.add_argument("--atol", type=float, default=5e-2)
    s_check.set_defaults(func=cmd_check)

    return p

def main():
    args = build_parser().parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
