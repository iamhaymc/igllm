# Local-weight benchmarks

These are **local zero-shot diagnostics, not official leaderboard scores**. A dry run measures nothing. N/A means not run or an operational failure, never zero accuracy.

- Generated (UTC): 2026-09-07T00:51:02.916413+00:00
- Local checkpoint: `/home/runner/work/igllm/igllm/model`; engine: `/home/runner/work/igllm/igllm/build/igllm`.
- Selection: first 5 eligible examples per benchmark in source order; MMMU cycles through alphabetically sorted subject configurations.
- Greedy decoding: temperature 0, seed 42, repetition penalty 1, 256 generated-token cap, 1 thread(s), 120s timeout per inference; chat framing enabled.
- Each example starts a fresh native process (checkpoint load included in runtime). No model downloads, generated-code execution, tools, browser, remote inference, or dataset scripts.
- Dataset access requires network/cache and may download large shards even for a small sample. HF revisions resolve to recorded commits; reproduce with --revision COMMIT and one --benchmarks selection. BFCL uses the fixed source commit below.

| Benchmark / local metric | Status | Correct / completed | Errors | Skipped | Accuracy |
|---|---|---:|---:|---:|---:|
| Reasoning: GSM8K numeric exact match | NOT RUN | 0 / 0 | 0 | 0 | N/A |
| Reasoning: MMLU-Pro choice accuracy | NOT RUN | 0 / 0 | 0 | 0 | N/A |
| Vision: MMMU multiple-choice accuracy | NOT RUN | 0 / 0 | 0 | 0 | N/A |
| Agentic proxy: BFCL v4 simple Python strict AST match | NOT RUN | 0 / 0 | 0 | 0 | N/A |

## Coverage and scoring limitations

- GSM8K: official test questions; local numeric exact match after `Final answer:` or `####` (or number-only output), ignoring decimal formatting and commas; no few-shot exemplars.
- MMLU-Pro: test questions; exact final option letter, no official five-shot CoT prompt.
- MMMU: validation **multiple-choice only**; open-ended items excluded and counted when encountered. Every non-null image slot is supplied in numbered order, including images referenced by options. Images precede text with explicit slot labels rather than native interleaving. No text-only fallback. A small sample does not cover all subjects; --limit 0 covers all eligible MC items, not open-ended ones.
- BFCL v4: **simple_python only**, single-turn, single-call, keyword/literal AST matching with reference alternatives and the empty-string omission sentinel. Strings/list order/types are strict (integer/float numeric equality allowed); argument order is ignored. Positional calls, expression evaluation, and full official normalization are not supported. This is a narrow function-calling proxy, **not** an autonomous-agent or full BFCL score.
- SWE-bench Verified, GAIA, WebArena and OSWorld require separate execution/browser/desktop environments and official harnesses; they are not measured here. No aggregate across these tasks.
- Tiny deterministic prefixes are smoke tests, not representative estimates. Token limits can truncate reasoning; malformed model answers count wrong. Operational/data errors make the entire affected benchmark N/A and the process exit nonzero. No retries or hidden exclusion of failures.
- Metadata hashes and weight sizes/mtimes identify local artifacts only approximately: weight contents are not hashed. Greedy results can still differ across hardware/compiler builds.

## Reproduction

```sh
python run.py build --only igllm
python -m pip install datasets==4.0.0 Pillow==12.3.0
python weights_bench.py --model model/ --limit 5
python weights_bench.py --model model/ --limit 0  # potentially very expensive
python weights_bench.py --dry-run  # no imports of ML packages or downloads
```

## Provenance

- gsm8k: [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k), config `main`, split `test`; {"requested_revision": "main"}.
- mmlu-pro: [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro), config `default`, split `test`; {"requested_revision": "main"}.
- mmmu: [MMMU/MMMU](https://huggingface.co/datasets/MMMU/MMMU), config `all subjects`, split `validation`; {"requested_revision": "main"}.
- bfcl: [ShishirPatil/gorilla](https://raw.githubusercontent.com/ShishirPatil/gorilla/6ea57973c7a6097fd7c5915698c54c17c5b1b6c8/berkeley-function-call-leaderboard/bfcl_eval/data/), config `simple_python`, split `BFCL v4`; {"requested_revision": "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"}.

Run metadata (JSON):

```json
{
  "arguments": {
    "benchmarks": [
      "gsm8k",
      "mmlu-pro",
      "mmmu",
      "bfcl"
    ],
    "dry_run": true,
    "engine": "/home/runner/work/igllm/igllm/build/igllm",
    "limit": 5,
    "max_tokens": 256,
    "model": "/home/runner/work/igllm/igllm/model",
    "output": "/home/runner/work/igllm/igllm/BENCHMARKS.md",
    "raw": false,
    "revision": "main",
    "seed": 42,
    "threads": 1,
    "timeout": 120
  },
  "harness_sha256": "d5cf25ca71ce3a3badd818e84148883791d4570f5c0796e32b2a192119142877",
  "packages": {
    "Pillow": "not installed",
    "datasets": "not installed",
    "huggingface_hub": "1.30.0"
  },
  "python": "3.12.3 (main, Jun 19 2026, 12:46:00) [GCC 13.3.0]"
}
```

## Per-example audit

Input SHA-256 covers the exact prompt, reference, and exported image hashes. Outputs below are escaped, untrusted model text, not instructions.

| Benchmark | Config / ID | Correct | Prediction / error | Seconds | Input SHA-256 |
|---|---|---|---|---:|---|
