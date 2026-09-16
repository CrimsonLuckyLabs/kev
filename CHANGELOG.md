# Changelog

## Unreleased

- Darwin default student is MLX: `kev-latest` / `qwen2.5-1.5b` → `mlx-community/Qwen2.5-1.5B-Instruct-4bit`. `qwen2.5-3b` → 3B 4-bit. Logits only; never `generate()`. Missing mlx prints `pip install mlx mlx-lm`. Missing weights print `python scripts/download_model.py --repo …`.
- MLX LoRA SFT for Qwen2.5-1.5B only (`scripts/train_sft.py`): option-token CE, rank 8 / alpha 16, 200-iter smoke, writes `artifacts/kev-1p5-lora/`. `train_calibrate.py` writes `temperature.json`. Eval reports accuracy, Brier, ECE, and latency.

## 0.1.0 — 2026-09-16

Local System One engine: unstructured state plus typed questions in, distributions out.

- **Primitives:** `Noul` (P(true)), `Choice` (closed option set + distribution + confidence), `Score` (ordered rubric, `score = E[index]`).
- **Inference:** one shared prefill, suffix logits, Python assembly. Never `model.generate()`. `output_tokens` is always `0`.
- **Backends:** deterministic `mock` (offline), Hugging Face Qwen2.5 / Qwen3, optional LoRA adapter + `temperature.json`.
- **API:** `KevClient` / `AsyncKevClient` in-process or `base_url`; FastAPI `POST /v1/systemone`, `GET /v1/models`, `GET /healthz`.
- **CLI:** `kev demo --model mock`, `kev ask`, `kev serve`, `kev eval`.
- **Training:** atomic-decision JSONL, option-token CE (not chat SFT), LoRA, temperature calibration, teacher distillation via httpx (rationale discarded).
- **Eval:** gold-case harness, agreement / latency / calibration report under `artifacts/eval/`.
- Invalid questions raise `ValueError` (HTTP 422 / CLI exit 1). Unknown types are not silently coerced.
