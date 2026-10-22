"""M0 acceptance check (agent.md Section 11): imports work, DEVICE set, Ollama
model answers a test prompt.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ROOT, env, load_config  # noqa: E402

REQUIRED = [
    ("torch", "__version__"),
    ("torch_geometric", "__version__"),
    ("networkx", "__version__"),
    ("sentence_transformers", "__version__"),
    ("datasets", "__version__"),
    ("psutil", "__version__"),
    ("numpy", "__version__"),
    ("scipy", "__version__"),
    ("yaml", "__version__"),
    ("sklearn", "__version__"),
]


def check_imports() -> tuple[bool, dict]:
    print("--- imports ---")
    versions, ok = {}, True
    for mod, attr in REQUIRED:
        try:
            m = importlib.import_module(mod)
            v = getattr(m, attr, "?")
            versions[mod] = v
            print(f"  OK   {mod:<22} {v}")
        except Exception as exc:
            ok = False
            versions[mod] = f"FAIL: {exc!r}"
            print(f"  FAIL {mod:<22} {exc!r}")
    # project modules
    for mod in ("src.schemas", "src.profiling", "src.ollama_client", "src.config"):
        try:
            importlib.import_module(mod)
            print(f"  OK   {mod}")
        except Exception as exc:
            ok = False
            versions[mod] = f"FAIL: {exc!r}"
            print(f"  FAIL {mod:<22} {exc!r}")
    return ok, versions


def check_mps_fallback() -> bool:
    import torch

    flag = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
    print("\n--- torch / device ---")
    print(f"  PYTORCH_ENABLE_MPS_FALLBACK = {flag}")
    print(f"  torch {torch.__version__}  mps.is_available()={torch.backends.mps.is_available()}"
          f"  mps.is_built()={torch.backends.mps.is_built()}")
    return flag == "1"


def check_ollama(tag: str) -> tuple[bool, dict]:
    from src.ollama_client import OllamaClient, OllamaError

    print("\n--- ollama ---")
    info = {"model_tag": tag}
    cli = OllamaClient(tag, num_ctx=2048, temperature=0.0)
    try:
        info["server_version"] = cli.version()
        print(f"  server version: {info['server_version']}")
        models = cli.list_models()
        info["models"] = models
        print(f"  models: {models}")
        cli.assert_model_present()
    except (OllamaError, Exception) as exc:
        print(f"  FAIL {exc!r}")
        info["error"] = repr(exc)
        return False, info

    prompt = "Answer with one word only. What is the capital of France?"
    t0 = time.perf_counter()
    text, usage, raw = cli.generate(prompt, fmt=None, num_predict=16)
    dt = time.perf_counter() - t0
    info.update({
        "test_prompt": prompt,
        "test_response": text.strip(),
        "latency_s": dt,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
    })
    print(f"  prompt   : {prompt}")
    print(f"  response : {text.strip()!r}")
    print(f"  latency  : {dt:.2f} s | tokens: {usage.prompt_tokens} prompt + {usage.completion_tokens} eval")
    ok = bool(text.strip()) and "paris" in text.lower()
    print(f"  answer contains 'Paris': {ok}")

    # JSON mode smoke test -- extraction depends on format="json" (Section 6.2)
    jtext, jusage, _ = cli.generate(
        'Return JSON only: {"triples": [["Paris", "capital of", "France"]]}. Echo it.',
        fmt="json", num_predict=64,
    )
    try:
        json.loads(jtext)
        print(f"  format=json parses: True")
        info["json_mode_ok"] = True
    except Exception as exc:
        print(f"  format=json parses: False ({exc})")
        info["json_mode_ok"] = False
        ok = False
    return ok, info


def check_embedder(model_name: str) -> tuple[bool, dict]:
    """Not an M0 acceptance item, but catches a bad transformers/ST pin before we
    freeze requirements.txt."""
    print("\n--- embedder smoke test ---")
    info = {"model": model_name}
    try:
        from sentence_transformers import SentenceTransformer

        t0 = time.perf_counter()
        m = SentenceTransformer(model_name, device="cpu")
        v = m.encode(["capital of France"], normalize_embeddings=True)
        info["load_s"] = time.perf_counter() - t0
        info["dim"] = int(v.shape[1])
        info["norm"] = float((v[0] ** 2).sum() ** 0.5)
        print(f"  loaded in {info['load_s']:.1f}s | dim={info['dim']} | ||v||={info['norm']:.4f}")
        ok = info["dim"] == 384 and abs(info["norm"] - 1.0) < 1e-3
        print(f"  dim==384 and normalized: {ok}")
        del m
        return ok, info
    except Exception as exc:
        print(f"  FAIL {exc!r}")
        info["error"] = repr(exc)
        return False, info


def check_profiling() -> tuple[bool, dict]:
    print("\n--- profiling (process tree + ollama RSS) ---")
    from src.profiling import StageProfiler, ollama_processes, tree_rss_mb, virtual_memory_snapshot

    n_oll = len(ollama_processes())
    with StageProfiler("m0_smoke", interval_s=0.1) as prof:
        with prof.item():
            time.sleep(0.35)
    s = prof.stats
    print(f"  ollama processes seen: {n_oll}")
    print(f"  tree RSS now: {tree_rss_mb():.1f} MB")
    print(f"  stage peak {s.peak_rss_mb:.1f} MB | baseline {s.baseline_rss_mb:.1f} MB "
          f"| delta {s.delta_rss_mb:.1f} MB | swap delta {s.swap_delta_mb:.1f} MB")
    print(f"  vm: {virtual_memory_snapshot()}")
    ok = s.peak_rss_mb > 0 and n_oll > 0 and prof.latency_summary()["n"] == 1
    print(f"  sampler works and sees ollama: {ok}")
    return ok, {"ollama_procs": n_oll, **s.as_dict()}


def run(cfg, args) -> int:
    report = {}
    ok_imp, report["imports"] = check_imports()
    ok_fb = check_mps_fallback()
    report["mps_fallback_set"] = ok_fb

    ok_oll, report["ollama"] = check_ollama(env("SLM_TAG"))
    ok_emb, report["embedder"] = check_embedder(env("EMBED_MODEL"))
    ok_prof, report["profiling"] = check_profiling()

    print("\n--- device benchmark (mps vs cpu) ---")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "bench_device.py")],
                       capture_output=False)
    ok_dev = r.returncode == 0
    cfg2 = load_config()
    device = cfg2.get("device")
    report["device"] = device
    bench_path = ROOT / "results" / "device_benchmark.json"
    if bench_path.exists():
        report["device_benchmark"] = json.loads(bench_path.read_text())

    checks = {
        "imports work": ok_imp,
        "PYTORCH_ENABLE_MPS_FALLBACK=1": ok_fb,
        "DEVICE set in config": bool(device) and ok_dev,
        "ollama model answers test prompt": ok_oll,
        "embedder loads (384-dim, normalized)": ok_emb,
        "process-tree RSS sampler sees ollama": ok_prof,
    }
    print("\n=== M0 ACCEPTANCE ===")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    report["acceptance"] = checks
    report["all_passed"] = all(checks.values())

    out = ROOT / "results" / "m0_report.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nreport -> {out.relative_to(ROOT)}")
    print(f"M0: {'PASS' if report['all_passed'] else 'FAIL'}")
    return 0 if report["all_passed"] else 1
