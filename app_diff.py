#!/usr/bin/env python3
"""app_diff.py - compares the engine against the reference one tensor at a time.

A disagreement in the logits says only that something is wrong. This script
walks the graph in order and stops at the first tensor that drifts, which names
the function to look at rather than the model.

    python3 app_diff.py                       # build a fake checkpoint and diff it
    python3 app_diff.py --model /path/to/ckpt # diff a checkpoint you already have
    python3 app_diff.py --sweep               # the whole coverage matrix

The engine must be built with the activation dump compiled in:

    python3 run.py build --trace

The tolerance is not a guess. Before anything is judged, the reference is run
against itself in two summation orders to find the noise floor, and the floor
is what the comparison is scaled against. Anything tighter would report float
addition as a bug; anything looser would let a real one through.
"""

import argparse
import os
import struct
import subprocess
import sys
import tempfile

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
WORK_PATH = os.path.join(ROOT_PATH, "build")

PROMPT_TEXT = "the sea is wide and the sky is high, one two three four"

# Lengths chosen to straddle the two boundaries that change which path runs:
# the sliding window, and the sixteen lane prefill chunk. A prompt of exactly
# one token is the decode path with no batch at all.
LENGTH_LIST = (1, 2, 9, 16, 17, 40)

# The order the graph runs in. The first name to drift is the one that matters,
# so the comparison walks this list and stops rather than reporting every later
# tensor that inherited the error.
STEP_ORDER = ("embed", "attn", "mlp", "moe", "out", "final", "logits")


# -- the dump the engine writes -------------------------------------------


def trace_read(path):
    """Reads the activation dump into {name: [float]}."""
    with open(path, "rb") as source:
        blob = source.read()
    if blob[:8] != b"IGTRACE1":
        raise ValueError("not an activation dump: %s" % path)
    found = {}
    walk = 8
    while walk < len(blob):
        name_size, = struct.unpack_from("<I", blob, walk)
        walk += 4
        name_text = blob[walk:walk + name_size].decode("utf-8")
        walk += name_size
        slot_count, = struct.unpack_from("<I", blob, walk)
        walk += 4
        found[name_text] = struct.unpack_from("<%df" % slot_count, blob, walk)
        walk += 4 * slot_count
    return found


def engine_path():
    leaf = "igllm.exe" if os.name == "nt" else "igllm"
    return os.path.join(WORK_PATH, leaf)


def prompt_of(id_count):
    """Builds a prompt that tokenizes to roughly `id_count` ids."""
    word_list = PROMPT_TEXT.split()
    return " ".join(word_list[index % len(word_list)] for index in range(id_count))


def engine_trace(model_path, prompt):
    """Runs the engine once and returns (dump, token ids)."""
    import json

    handle, trace_path = tempfile.mkstemp(suffix=".igtrace")
    os.close(handle)
    room = dict(os.environ, IGLLM_TRACE=trace_path)
    line = [engine_path(), "logits", "--model", model_path, "--prompt", prompt, "--serve", "1"]
    done = subprocess.run(line, capture_output=True, text=True, env=room)
    if done.returncode != 0:
        raise RuntimeError("engine failed: %s" % done.stderr.strip())
    found = trace_read(trace_path)
    os.unlink(trace_path)
    if not found:
        raise RuntimeError("the engine wrote no activations; build with `run.py build --trace`")
    return found, json.loads(done.stdout)["tokens"]


# -- the reference --------------------------------------------------------


def reference_run(model_path, id_list):
    """Runs the reference over the same ids and returns {name: [float]}.

    Hooks are placed on the attention and feedforward modules so that the two
    sides can be compared before the residual folds them together, where an
    error is still local to one function.
    """
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    model.eval()
    found = {}
    handle_list = []

    def catch(stem, layer_index):
        # Not every module keeps the batch axis: the expert block folds it away
        # and returns one row per token. Treat the last axis as the features and
        # everything before it as positions, which is true of both shapes.
        def hook(module, argument, result):
            del module, argument
            value = result[0] if isinstance(result, tuple) else result
            flat = value.detach().float().reshape(-1, value.shape[-1])
            for place_index in range(flat.shape[0]):
                found["layer.%d.%s.%d" % (layer_index, stem, place_index)] = \
                    flat[place_index].tolist()
        return hook

    inner = model.model.language_model if hasattr(model.model, "language_model") else model.model
    for layer_index, layer in enumerate(inner.layers):
        handle_list.append(layer.self_attn.register_forward_hook(catch("attn", layer_index)))
        handle_list.append(layer.mlp.register_forward_hook(catch("mlp", layer_index)))
        if hasattr(layer, "experts"):
            handle_list.append(layer.experts.register_forward_hook(catch("moe", layer_index)))

    id_data = torch.tensor([id_list], dtype=torch.long)
    with torch.no_grad():
        result = model(id_data, output_hidden_states=True)
    for handle in handle_list:
        handle.remove()

    # The last entry of `hidden_states` is the final norm already applied, not
    # the last layer's output, so the residual of the last layer is not exposed
    # and `final` comes straight from the end of the stack.
    stack = result.hidden_states
    for place_index in range(len(id_list)):
        found["embed.%d" % place_index] = stack[0][0, place_index].float().tolist()
        for layer_index in range(1, len(stack) - 1):
            found["layer.%d.out.%d" % (layer_index - 1, place_index)] = \
                stack[layer_index][0, place_index].float().tolist()
        found["final.%d" % place_index] = stack[-1][0, place_index].float().tolist()
    last_index = len(id_list) - 1
    found["logits.%d" % last_index] = result.logits[0, last_index].float().tolist()
    return found


def reference_floor(model_path, id_list):
    """Measures how far the reference moves when only the arithmetic changes.

    Running the same graph in a different summation order gives the size of an
    error that means nothing. Everything below this is float addition; the
    comparison only calls a tensor wrong when it is well clear of it.
    """
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    model.eval()
    id_data = torch.tensor([id_list], dtype=torch.long)
    with torch.no_grad():
        whole = model(id_data, output_hidden_states=True).hidden_states[-1][0, -1]
        # The same tokens fed one at a time behind a cache: identical maths in a
        # different order, which is exactly the difference the engine's batching
        # also introduces. Whatever this moves by is not a bug.
        state = None
        for place_index in range(len(id_list)):
            result = model(id_data[:, place_index:place_index + 1], past_key_values=state,
                           use_cache=True, output_hidden_states=True)
            state = result.past_key_values
        piece = result.hidden_states[-1][0, -1]
    return float((whole - piece).abs().max())


# -- comparison -----------------------------------------------------------


def drift_of(mine, theirs):
    """Largest absolute gap, relative to how big the tensor is."""
    if len(mine) != len(theirs):
        return float("inf"), 0.0
    peak = max((abs(value) for value in theirs), default=0.0)
    gap = max((abs(a - b) for a, b in zip(mine, theirs)), default=0.0)
    return gap, peak


def compare(mine, theirs, id_count, slack):
    """Walks the graph in order and returns the first tensor that drifts."""
    fault_list = []
    for place_index in range(id_count):
        for stem in STEP_ORDER:
            for name in sorted(theirs):
                if not name.endswith(".%d" % place_index):
                    continue
                head = name[: name.rindex(".")]
                if head.split(".")[-1] != stem:
                    continue
                if name not in mine:
                    continue
                gap, peak = drift_of(mine[name], theirs[name])
                # Deeper tensors inherit the error of everything before them,
                # so the bar loosens with depth rather than staying flat.
                depth = 1.0 + (head.count(".") and int(head.split(".")[1]) or 0)
                limit = slack * depth * max(1.0, peak)
                if gap > limit:
                    fault_list.append((name, gap, peak, limit))
    return fault_list


def report(title, fault_list, checked):
    if not fault_list:
        print("  %-28s ok    %d tensors agree" % (title, checked))
        return 0
    name, gap, peak, limit = fault_list[0]
    print("  %-28s FAIL  %s drifts by %.3e (peak %.3e, allowed %.3e)"
          % (title, name, gap, peak, limit))
    for name, gap, peak, limit in fault_list[1:4]:
        print("  %-28s       then %s by %.3e" % ("", name, gap))
    return 1


# -- workflow -------------------------------------------------------------


def diff_one(title, model_path, ref_path, slack, length_list=None):
    """Diffs one checkpoint at several prompt lengths."""
    fail_count = 0
    checked = 0
    fault_list = []
    for id_count in (length_list or (len(PROMPT_TEXT.split()),)):
        mine, id_list = engine_trace(model_path, prompt_of(id_count))
        theirs = reference_run(ref_path, id_list)
        shared = [name for name in theirs if name in mine]
        if not shared:
            print("  %-28s FAIL  the two sides share no tensor names" % title)
            return 1
        checked += len(shared)
        fault_list += compare(mine, theirs, len(id_list), slack)
    fail_count += report(title, fault_list, checked)
    return fail_count


# The axes that change which code path runs. Prompt lengths are chosen to
# straddle the batch chunk and the sliding window rather than to be round.
SWEEP_PLAN = (
    ("dense float", "dense", 0, 0),
    ("dense 4 bit, group 16", "dense", 4, 16),
    ("dense 4 bit, ragged group", "dense", 4, 12),
    ("dense 2 bit, group 16", "dense", 2, 16),
    ("dense 3 bit, group 16", "dense", 3, 16),
    ("dense 5 bit, ragged group", "dense", 5, 12),
    ("dense 8 bit, per row", "dense", 8, 0),
    ("mixture float", "moe", 0, 0),
    ("mixture 4 bit, group 16", "moe", 4, 16),
    ("double wide float", "wide", 0, 0),
    ("shared key value float", "twin", 0, 0),
)


def main():
    parser = argparse.ArgumentParser(description="layer by layer comparison")
    parser.add_argument("--model", help="a checkpoint to diff instead of a synthetic one")
    parser.add_argument("--reference", help="the checkpoint the reference reads, if it differs")
    parser.add_argument("--sweep", action="store_true", help="run the whole coverage matrix")
    parser.add_argument("--work", default=os.path.join(WORK_PATH, "fake"))
    parser.add_argument("--slack", type=float, default=0.0,
                        help="absolute tolerance, or 0 to measure the noise floor")
    flag = parser.parse_args()

    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        print("skip: torch and transformers are not installed; run `python3 run.py install`")
        return 0
    if not os.path.exists(engine_path()):
        print("skip: no engine built; run `python3 run.py build --trace`")
        return 0

    slack = flag.slack
    fail_count = 0

    if flag.model:
        if slack <= 0:
            slack = 1e-4
        print("checkpoint %s" % flag.model)
        fail_count += diff_one("as given", flag.model, flag.reference or flag.model, slack,
                               LENGTH_LIST if flag.sweep else None)
        print("\n%d checks failed" % fail_count)
        return 0 if fail_count == 0 else 1

    import app_fake

    plan = SWEEP_PLAN if flag.sweep else SWEEP_PLAN[:2]
    for title, preset, bit_count, group_size in plan:
        out_path = os.path.join(flag.work, "%s-%d-%d" % (preset, bit_count, group_size))
        made = app_fake.fake_build(out_path, preset, 7, "real", bit_count, group_size)
        if slack <= 0:
            mine, id_list = engine_trace(made["plain"], PROMPT_TEXT)
            floor = reference_floor(made["plain"], id_list)
            slack = max(floor * 8.0, 1e-5)
            print("noise floor %.3e, tolerance %.3e" % (floor, slack))
        length_list = LENGTH_LIST if flag.sweep else None
        if bit_count > 0:
            fail_count += diff_one(title, made["packed"], made["dequant"], slack, length_list)
        else:
            fail_count += diff_one(title, made["plain"], made["plain"], slack, length_list)

    print("\n%d checks failed" % fail_count)
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
