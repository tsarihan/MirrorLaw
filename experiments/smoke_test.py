#!/usr/bin/env python3
"""
Anvil H100 smoke test — run this in gpu-debug BEFORE spending real GPU hours.
Verifies: torch sees CUDA, the device is an H100, bf16 works, a real matmul runs,
and (optionally, with --model) a small model loads and generates.

    apptainer exec --nv pytorch.sif python smoke_test.py
    apptainer exec --nv vllm.sif    python smoke_test.py --model Qwen/Qwen2.5-0.5B-Instruct
"""
import argparse
import sys
import time

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="optional HF id / local path to load + generate")
    ap.add_argument("--expect", default="H100", help="substring expected in the device name")
    args = ap.parse_args()

    print("=" * 56)
    print("ANVIL GPU SMOKE TEST")
    print("=" * 56)

    # 1. torch + CUDA
    try:
        import torch
    except Exception as e:  # noqa
        check("import torch", False, repr(e))
        _summary()
        return
    check("import torch", True, f"torch {torch.__version__}")

    cuda_ok = check("cuda available", torch.cuda.is_available())
    if not cuda_ok:
        print("\n  -> No CUDA. Are you in an --nv apptainer on a gpu node? (not a login node)")
        _summary()
        return

    n = torch.cuda.device_count()
    check("gpu count", n > 0, f"{n} visible")
    dev_name = torch.cuda.get_device_name(0)
    check(f"device is {args.expect}", args.expect.lower() in dev_name.lower(), dev_name)

    # 2. bf16 + a real matmul on-device (H100 should be fast)
    try:
        x = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            y = x @ x
        torch.cuda.synchronize()
        dt = (time.time() - t0) / 10
        tflops = (2 * 8192 ** 3) / dt / 1e12
        check("bf16 matmul", True, f"{dt*1e3:.1f} ms/iter (~{tflops:.0f} TFLOP/s)")
        del x, y
        torch.cuda.empty_cache()
    except Exception as e:  # noqa
        check("bf16 matmul", False, repr(e))

    # 3. memory visible
    try:
        free, total = torch.cuda.mem_get_info()
        check("gpu memory", total / 1e9 > 60, f"{total/1e9:.0f} GB total ({free/1e9:.0f} free)")
    except Exception as e:  # noqa
        check("gpu memory", False, repr(e))

    # 4. optional: load a model + generate
    if args.model:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(args.model)
            model = AutoModelForCausalLM.from_pretrained(
                args.model, torch_dtype=torch.bfloat16, device_map="cuda"
            )
            ids = tok("Patent claim 1. A system comprising", return_tensors="pt").to("cuda")
            out = model.generate(**ids, max_new_tokens=16)
            txt = tok.decode(out[0], skip_special_tokens=True)
            check("model load + generate", True, f"{args.model} -> {len(txt)} chars")
        except Exception as e:  # noqa
            check("model load + generate", False, repr(e))

    _summary()


def _summary():
    print("-" * 56)
    ok = all(r[1] for r in results)
    print(f"  OVERALL: {'PASS -> safe to run real GPU jobs' if ok else 'FAIL -> fix before spending GPU hours'}")
    print("=" * 56)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
