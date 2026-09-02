#!/usr/bin/env python3
"""app_test.py - compares the C engine against the transformers reference.

The script drives the built `igllm` binary and the Hugging Face implementation
over the same prompts, then reports three things:

  accuracy   agreement of the next token distribution (top-k overlap, rank one
             match, and the largest absolute logit gap)
  behaviour  agreement of the greedy continuation, token for token
  speed      decode tokens per second of each side, and their ratio

The checkpoint is never bundled. When it is absent, or when torch and
transformers are not installed, the script reports what it skipped and exits
zero so that it stays usable inside a build pipeline.

    python3 app_test.py --model /path/to/gemma-4-E2B-it-qat
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
WORK_PATH = os.path.join(ROOT_PATH, "build")

PROMPT_LIST = [
    "Hello!",
    "Write one sentence about the sea.",
    "What is the capital of France?",
    "Explain gravity to a child in two sentences.",
]

TOP_COUNT = 16
SERVE_COUNT = 32
LOGIT_SLACK = 0.35
RANK_SHARE = 0.75


# -- engine side ----------------------------------------------------------


def engine_path():
    leaf = "igllm.exe" if os.name == "nt" else "igllm"
    return os.path.join(WORK_PATH, leaf)


def engine_build():
    line = [sys.executable, os.path.join(ROOT_PATH, "run.py"), "build"]
    return subprocess.call(line) == 0


def engine_call(model_path, task, prompt, extra=None):
    line = [engine_path(), task, "--model", model_path, "--prompt", prompt]
    if extra:
        line += extra
    done = subprocess.run(line, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError("engine %s failed: %s" % (task, done.stderr.strip()))
    return done.stdout, done.stderr


def engine_logits(model_path, prompt):
    out_text, _ = engine_call(model_path, "logits", prompt, ["--serve", str(TOP_COUNT)])
    return json.loads(out_text)


def engine_serve(model_path, prompt):
    from_time = time.time()
    out_text, err_text = engine_call(
        model_path, "chat", prompt, ["--serve", str(SERVE_COUNT), "--heat", "0"]
    )
    rate_match = re.search(r"decode\s+([0-9.]+) tok/s", err_text)
    return {
        "text": out_text.rstrip("\n"),
        "rate": float(rate_match.group(1)) if rate_match else 0.0,
        "wall": time.time() - from_time,
    }


# -- reference side -------------------------------------------------------


def reference_open(model_path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    book = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    model.eval()
    return book, model


def reference_frame(book, prompt):
    turn_list = [{"role": "user", "content": prompt}]
    if getattr(book, "chat_template", None):
        return book.apply_chat_template(turn_list, tokenize=False, add_generation_prompt=True)
    return prompt


def reference_logits(book, model, prompt):
    import torch

    text = reference_frame(book, prompt)
    piece = book(text, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        value_list = model(**piece).logits[0, -1].float()
    order = torch.topk(value_list, TOP_COUNT)
    return {
        "tokens": piece["input_ids"][0].tolist(),
        "top": [
            {"id": int(slot), "logit": float(value)}
            for slot, value in zip(order.indices.tolist(), order.values.tolist())
        ],
    }


def reference_serve(book, model, prompt):
    import torch

    text = reference_frame(book, prompt)
    piece = book(text, return_tensors="pt", add_special_tokens=False)
    from_time = time.time()
    with torch.no_grad():
        grown = model.generate(**piece, max_new_tokens=SERVE_COUNT, do_sample=False)
    spent = time.time() - from_time
    fresh = grown[0][piece["input_ids"].shape[1]:]
    return {
        "text": book.decode(fresh, skip_special_tokens=True),
        "rate": len(fresh) / spent if spent > 0 else 0.0,
        "wall": spent,
    }


# -- comparison -----------------------------------------------------------


def compare_logits(mine, theirs):
    my_top = [entry["id"] for entry in mine["top"]]
    their_top = [entry["id"] for entry in theirs["top"]]
    my_value = {entry["id"]: entry["logit"] for entry in mine["top"]}
    shared = set(my_top) & set(their_top)
    gap_value = max(
        (abs(my_value[entry["id"]] - entry["logit"]) for entry in theirs["top"] if entry["id"] in my_value),
        default=float("inf"),
    )
    return {
        "lead_match": bool(my_top and their_top and my_top[0] == their_top[0]),
        "share": len(shared) / float(len(their_top)) if their_top else 0.0,
        "gap": gap_value,
        "token_match": mine["tokens"] == theirs["tokens"],
    }


def report_line(name, truth_flag, detail):
    print("  %-12s %-4s %s" % (name, "ok" if truth_flag else "FAIL", detail))


def main():
    parser = argparse.ArgumentParser(description="igllm parity and speed comparison")
    parser.add_argument("--model", default=os.environ.get("IGLLM_MODEL"),
                        help="checkpoint folder in huggingface layout")
    parser.add_argument("--prompts", nargs="*", default=PROMPT_LIST)
    parser.add_argument("--skip-speed", action="store_true")
    flag = parser.parse_args()

    if not flag.model or not os.path.isdir(flag.model):
        print("skip: no checkpoint folder given; pass --model or set IGLLM_MODEL")
        return 0
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        print("skip: torch and transformers are not installed; run `python3 run.py install`")
        return 0
    if not engine_build():
        print("fail: the engine did not build")
        return 1

    print("loading the reference implementation ...")
    book, model = reference_open(flag.model)

    fail_count = 0
    my_rate_list = []
    their_rate_list = []

    for prompt in flag.prompts:
        print("\nprompt: %r" % prompt)
        mine = engine_logits(flag.model, prompt)
        theirs = reference_logits(book, model, prompt)
        verdict = compare_logits(mine, theirs)

        report_line("tokens", verdict["token_match"],
                    "%d ids from the engine, %d from the reference"
                    % (len(mine["tokens"]), len(theirs["tokens"])))
        report_line("rank one", verdict["lead_match"],
                    "engine %d, reference %d" % (mine["top"][0]["id"], theirs["top"][0]["id"]))
        report_line("top %d" % TOP_COUNT, verdict["share"] >= RANK_SHARE,
                    "%.0f%% of the reference ids are shared" % (100.0 * verdict["share"]))
        report_line("logit gap", verdict["gap"] <= LOGIT_SLACK,
                    "largest absolute gap %.4f" % verdict["gap"])
        fail_count += sum(
            0 if truth else 1
            for truth in (verdict["lead_match"], verdict["share"] >= RANK_SHARE,
                          verdict["gap"] <= LOGIT_SLACK)
        )

        if not flag.skip_speed:
            my_run = engine_serve(flag.model, prompt)
            their_run = reference_serve(book, model, prompt)
            my_rate_list.append(my_run["rate"])
            their_rate_list.append(their_run["rate"])
            same_flag = my_run["text"].strip() == their_run["text"].strip()
            report_line("greedy", same_flag,
                        "engine %r" % my_run["text"][:60] if same_flag
                        else "engine %r vs reference %r"
                             % (my_run["text"][:60], their_run["text"][:60]))
            report_line("decode", my_run["rate"] >= their_run["rate"],
                        "engine %.2f tok/s, reference %.2f tok/s"
                        % (my_run["rate"], their_run["rate"]))
            fail_count += 0 if same_flag else 1

    if my_rate_list and their_rate_list:
        my_mean = sum(my_rate_list) / len(my_rate_list)
        their_mean = sum(their_rate_list) / len(their_rate_list)
        share = my_mean / their_mean if their_mean > 0 else 0.0
        print("\nthroughput: engine %.2f tok/s, reference %.2f tok/s, ratio %.2fx"
              % (my_mean, their_mean, share))

    print("\n%d checks failed" % fail_count)
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
