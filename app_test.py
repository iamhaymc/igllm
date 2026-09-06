#!/usr/bin/env python3
"""app_test.py - compares the C engine against the transformers reference.

The script drives the built `igllm` binary and the Hugging Face implementation
over the same prompts, then reports three things:

  accuracy   agreement of the next token distribution (top-k overlap, rank one
             match, and the largest absolute logit gap, against how far the
             reference moves when only its own summation order changes)
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

# How much further than the reference's own movement the engine may go. Whether
# a rounding lands on a half step is a lottery, so what the reference draws on
# one prompt is an estimate of the movement and not a bound on it.
FLOOR_SHARE = 2.0


# -- engine side ----------------------------------------------------------


def engine_path():
    leaf = "igllm.exe" if os.name == "nt" else "igllm"
    return os.path.join(WORK_PATH, leaf)


def engine_build(build_flags=()):
    """Builds the engine, with whatever build flags the caller was given.

    The flags matter here rather than being a convenience: this rebuild
    overwrites whatever is in `build/`, so without them a `check --tuned` or a
    `check --wide` compared a default build and said nothing about the one
    asked for."""
    line = [sys.executable, os.path.join(ROOT_PATH, "run.py"), "build"] + list(build_flags)
    return subprocess.call(line) == 0


def engine_call(model_path, task, prompt, extra=None):
    line = [engine_path(), task, "--model", model_path, "--prompt", prompt]
    if extra:
        line += extra
    done = subprocess.run(line, capture_output=True, text=True, encoding="utf-8",
                          errors="replace")
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


def reference_steps(book, model, prompt):
    """The same distribution with the tokens fed one at a time behind a cache.

    This is identical maths in a different summation order, so whatever it
    moves by is not something the engine can be asked to avoid. On a checkpoint
    that rounds its activations onto a static grid the amount is not small: a
    sum landing on a half step falls one way in one order and the other way in
    another, which turns a last-bit disagreement into a whole step and then
    compounds it through the stack.
    """
    import torch

    text = reference_frame(book, prompt)
    piece = book(text, return_tensors="pt", add_special_tokens=False)
    id_data = piece["input_ids"]
    state = None
    with torch.no_grad():
        for place_index in range(id_data.shape[1]):
            result = model(id_data[:, place_index:place_index + 1], past_key_values=state,
                           use_cache=True)
            state = result.past_key_values
        value_list = result.logits[0, -1].float()
    order = torch.topk(value_list, TOP_COUNT)
    return {
        "tokens": id_data[0].tolist(),
        "top": [
            {"id": int(slot), "logit": float(value)}
            for slot, value in zip(order.indices.tolist(), order.values.tolist())
        ],
    }


def reference_split(book, model, prompt):
    """The same greedy continuation with the prompt primed one token at a time.

    `generate` reads the whole prompt in one forward and decodes from there;
    this primes the cache token by token first. Identical maths in another
    summation order, so where these two continuations part is where the
    checkpoint stops determining the answer and no engine can be asked to
    follow further.
    """
    import torch

    stop_value = getattr(model.generation_config, "eos_token_id", None)
    if isinstance(stop_value, (list, tuple)):
        stop_list = set(stop_value)
    else:
        stop_list = {stop_value} if stop_value is not None else set()

    text = reference_frame(book, prompt)
    piece = book(text, return_tensors="pt", add_special_tokens=False)
    id_data = piece["input_ids"]
    fresh_list = []
    state = None
    with torch.no_grad():
        for place_index in range(id_data.shape[1]):
            result = model(id_data[:, place_index:place_index + 1], past_key_values=state,
                           use_cache=True)
            state = result.past_key_values
        while len(fresh_list) < SERVE_COUNT:
            pick_id = int(result.logits[0, -1].argmax())
            if pick_id in stop_list:
                break
            fresh_list.append(pick_id)
            result = model(torch.tensor([[pick_id]], dtype=torch.long), past_key_values=state,
                           use_cache=True)
            state = result.past_key_values
    return book.decode(fresh_list, skip_special_tokens=True)


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


def share_head(left_text, right_text):
    """How many leading characters two answers have in common."""
    share_count = 0
    for left_char, right_char in zip(left_text, right_text):
        if left_char != right_char:
            break
        share_count += 1
    return share_count


def report_line(name, truth_flag, detail):
    print("  %-12s %-4s %s" % (name, "ok" if truth_flag else "FAIL", detail))


def main():
    parser = argparse.ArgumentParser(description="igllm parity and speed comparison")
    parser.add_argument("--model", default=os.environ.get("IGLLM_MODEL"),
                        help="checkpoint folder in huggingface layout")
    parser.add_argument("--prompts", nargs="*", default=PROMPT_LIST)
    parser.add_argument("--skip-speed", action="store_true")
    parser.add_argument("--build", action="append", default=[],
                        help="a flag to pass on to `run.py build`, repeatable")
    flag = parser.parse_args()

    # The model answers in whatever alphabet it likes, and a Windows console is
    # not usually in one of them. Report what cannot be encoded rather than
    # dying in the middle of a comparison.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    if not flag.model or not os.path.isdir(flag.model):
        print("skip: no checkpoint folder given; pass --model or set IGLLM_MODEL")
        return 0
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        print("skip: torch and transformers are not installed; run `python3 run.py install`")
        return 0
    if not engine_build(flag.build):
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
        # The bar for the logits is measured, not assumed: whatever the
        # reference fails to reproduce of itself, the engine is allowed too.
        floor_value = compare_logits(reference_steps(book, model, prompt), theirs)["gap"]
        if floor_value != floor_value or floor_value == float("inf"):
            floor_value = 0.0
        slack_value = max(LOGIT_SLACK, floor_value * FLOOR_SHARE)
        verdict = compare_logits(mine, theirs)

        report_line("tokens", verdict["token_match"],
                    "%d ids from the engine, %d from the reference"
                    % (len(mine["tokens"]), len(theirs["tokens"])))
        report_line("rank one", verdict["lead_match"],
                    "engine %d, reference %d" % (mine["top"][0]["id"], theirs["top"][0]["id"]))
        report_line("top %d" % TOP_COUNT, verdict["share"] >= RANK_SHARE,
                    "%.0f%% of the reference ids are shared" % (100.0 * verdict["share"]))
        report_line("logit gap", verdict["gap"] <= slack_value,
                    "largest absolute gap %.4f, the reference moves %.4f against itself"
                    % (verdict["gap"], floor_value))
        fail_count += sum(
            0 if truth else 1
            for truth in (verdict["lead_match"], verdict["share"] >= RANK_SHARE,
                          verdict["gap"] <= slack_value)
        )

        if not flag.skip_speed:
            my_run = engine_serve(flag.model, prompt)
            their_run = reference_serve(book, model, prompt)
            my_rate_list.append(my_run["rate"])
            their_rate_list.append(their_run["rate"])
            my_text = my_run["text"].strip()
            their_text = their_run["text"].strip()
            same_flag = my_text == their_text
            if same_flag:
                report_line("greedy", True, "engine %r" % my_text[:60])
            else:
                # Only when they part is it worth paying for the calibration:
                # the reference primed one token at a time parts from itself
                # somewhere too, and that is how far agreement can be asked for.
                my_share = share_head(my_text, their_text)
                their_share = share_head(reference_split(book, model, prompt).strip(), their_text)
                # Where a greedy chain parts is one draw from the same lottery
                # on both sides, so the engine is allowed to follow half as far
                # as the reference follows itself before it is called wrong.
                same_flag = my_share * FLOOR_SHARE >= their_share
                report_line("greedy", same_flag,
                            "engine follows for %d characters, the reference itself for %d: %r"
                            % (my_share, their_share, my_text[:60]))
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
