# igllm

A dependency-free inference engine for **Gemma 4 E2B IT QAT**, written in pure C11.

The engine loads a Hugging Face checkpoint directly — `config.json`,
`model.safetensors`, `tokenizer.json` — with no conversion step, no third
party library, and no build system beyond a C compiler.

- **pure C11** — the whole engine is one translation unit, `app_core.c`
- **no dependencies** — only the C standard library and the host thread API
- **cross-platform** — POSIX and Windows, x86-64 and ARM, with a scalar fallback
- **zero-copy weights** — packed quantized tensors are read straight from the
  memory-mapped checkpoint
- **accelerator ready** — every kernel is reached through a backend table of
  function pointers
- **batched prefill** — prompt tokens run in lanes, so a projection is a
  matrix product rather than a matrix-vector product per token
- **mixture-of-experts** — routed blocks run beside the shared expert, with
  the stacked expert weights sliced as views

## Quickstart

```sh
python3 run.py build                       # build the cli and the tests
python3 run.py test                        # build, then run the unit tests
python3 run.py run -- chat \
    --model /path/to/gemma-4-E2B-it-qat \
    --prompt "Explain gravity to a child."
```

The checkpoint folder is the one produced by
`huggingface-cli download google/gemma-4-E2B-it-qat-mobile-transformers`.
Nothing is downloaded by the engine itself.

## Tasks

| task       | purpose                                       |
| ---------- | --------------------------------------------- |
| `chat`     | one instruction-tuned turn, chat framed       |
| `complete` | raw continuation of the prompt text           |
| `bench`    | timed prefill and decode report               |
| `tokens`   | print the token ids of the prompt             |
| `logits`   | print the next token distribution as json     |
| `probe`    | print the resolved model shape                |

Common flags: `--model`, `--prompt`, `--serve`, `--threads`, `--window`,
`--heat`, `--top-k`, `--top-p`, `--echo-penalty`, `--seed`, `--raw`,
`--verbose`. Run `igllm --help` for the full list.

## Workflows

| command                       | effect                                        |
| ----------------------------- | --------------------------------------------- |
| `python3 run.py install`      | install the python packages parity work needs |
| `python3 run.py build`        | compile `igllm` and `igllm_test`              |
| `python3 run.py test`         | build, then run the unit tests                |
| `python3 run.py check --model <folder>` | test, then compare to the reference |
| `python3 run.py run -- <args>`| build, then run the cli                       |
| `python3 run.py clean`        | remove build products                         |

Add `--debug` for an unoptimized build with the address and behaviour
sanitizers, or `--tuned` to allow host specific instructions.

## Files

| file          | purpose                                        |
| ------------- | ---------------------------------------------- |
| `app_core.c`  | the engine, public interface and all ten layers |
| `app_main.c`  | the command line front end                     |
| `app_test.c`  | the unit tests                                 |
| `app_test.py` | comparison against the transformers reference  |
| `run.py`      | install, build, test, run workflows            |
| `GUIDE.md`    | a complete tour of the implementation          |
| `CHANGES.md`  | development progress and rationale             |
| `TODO.md`     | open development tasks                         |

`GUIDE.md` is the place to start if you intend to read or extend the code.

## Status

The engine builds clean and passes its unit tests on POSIX and Windows, and
under the address and behaviour sanitizers. Text generation, mixture-of-experts
blocks, and batched prefill are implemented; the vision and audio towers are
not, so the engine is text-only today. Numerical parity against the reference
has **not** been measured yet, because the checkpoint was unreachable from the
development sandbox; see `TODO.md`.
