#!/usr/bin/env python3
"""app_diff.py - compares the engine against the reference one tensor at a time.

A disagreement in the logits says only that something is wrong. This script
walks the graph in order and stops at the first tensor that drifts, which names
the function to look at rather than the model.

    python3 app_diff.py                       # build a fake checkpoint and diff it
    python3 app_diff.py --model /path/to/ckpt # diff a checkpoint you already have
    python3 app_diff.py --sweep               # the whole coverage matrix
    python3 app_diff.py --media               # the vision and audio towers
    python3 app_diff.py --media --model DIR   # the towers of a checkpoint you have
    python3 app_diff.py --seam                # the join between the two, end to end
    python3 app_diff.py --seam --model DIR    # the same join on a checkpoint you have

The engine must be built with the activation dump compiled in:

    python3 run.py build --trace

The tolerance is not a guess. Before anything is judged, the reference is run
against itself in two summation orders to find the noise floor, and the floor
is what the comparison is scaled against. Anything tighter would report float
addition as a bug; anything looser would let a real one through.

On a checkpoint that carries static activation ranges there is no one answer
to agree with. Every projection rounds what it reads and what it writes onto a
grid, so a sum that lands on a half step falls one way in single precision and
the other way in double, and one flipped step at one layer is a different
residual for every layer after it. The reference does this to itself: the same
graph in double precision agrees with single to a part in ten million through
the first layers, moves by whole steps from the middle of the stack, and by
whole units in the logits. Nothing that does not reproduce the reference's
summation order bit for bit can do better. So a real checkpoint is judged
differently from a synthetic one: how deep each side stays exact is reported,
the embedding and the first layer have to agree outright, and the end of the
stack is held to how far the reference moves there — over the whole run rather
than one prompt at a time, because whether a given perturbation lands on a
half step is a lottery, and a prompt where the reference happens to flip
nothing says nothing about the engine.

`--media` walks the two towers instead. Each is built from the reference's own
module and loaded with the checkpoint's own weights, which works even for the
shipped export: a tower is a couple of hundred megabytes and the reference gives
each one a `base_model_prefix` so it can be built alone, which is quicker and
lighter than standing the whole model up to look at one encoder.

Both sides start from the rows the engine says it read — the normalized patches,
or the mel frames — because the png reader, the resize and the filterbank are
each held against an independent definition in `app_test.c`, and what is in
question here is the tower above them.

`--seam` walks the join instead: the ids the reference's processor lays down
around a run of soft tokens, and the distribution the whole graph reaches over
them. Both halves run against the shipped export. The ids need no weights at
all, and the graph needs the whole model, which the export gives: its weights
stay packed and are decoded per forward, so it loads in about two and a third
gigabytes. `app_fake.whole_build` writes a synthetic one as well, which runs in
seconds where the real one takes minutes.
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

# How much further than the reference's own movement the engine may go at the
# end of the stack before it is called wrong.
FLOOR_SHARE = 2.0

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
    line = [engine_path(), "logits", "--model", model_path, "--prompt", prompt, "--serve", "1",
            "--raw"]
    done = subprocess.run(line, capture_output=True, text=True, env=room)
    if done.returncode != 0:
        raise RuntimeError("engine failed: %s" % done.stderr.strip())
    found = trace_read(trace_path)
    os.unlink(trace_path)
    if not found:
        raise RuntimeError("the engine wrote no activations; build with `run.py build --trace`")
    return found, json.loads(done.stdout)["tokens"]


# -- the reference --------------------------------------------------------


def reference_note(found, result, place_count):
    """Records the residual stream of one forward."""
    # The last entry of `hidden_states` is the final norm already applied, not
    # the last layer's output, so the residual of the last layer is not exposed
    # and `final` comes straight from the end of the stack.
    stack = result.hidden_states
    for place_index in range(place_count):
        found["embed.%d" % place_index] = stack[0][0, place_index].float().tolist()
        for layer_index in range(1, len(stack) - 1):
            found["layer.%d.out.%d" % (layer_index - 1, place_index)] = \
                stack[layer_index][0, place_index].float().tolist()
        found["final.%d" % place_index] = stack[-1][0, place_index].float().tolist()
    last_index = place_count - 1
    found["logits.%d" % last_index] = result.logits[0, last_index].float().tolist()


def reference_look(model_path, id_list, want_kind):
    """Runs the reference once in one precision and returns {name: [float]}.

    Hooks are placed on the attention and feedforward modules so that the two
    sides can be compared before the residual folds them together, where an
    error is still local to one function.
    """
    import gc

    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=want_kind)
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
    reference_note(found, result, len(id_list))
    # The caller runs this twice over, so the first model has to be gone before
    # the second is built: a forward materializes the whole embedding table to
    # read one row of it, which is 1.6 GiB single and 3.2 double.
    del model, result, handle_list
    gc.collect()
    return found


def reference_run(model_path, id_list, want_floor=False):
    """Runs the reference over the same ids and returns {name: [float]}.

    With `want_floor`, the same graph is run a second time in double precision
    and that second dictionary is returned beside the first. Wherever the two
    disagree, the tensor turns on the last bits of a float sum, and neither
    single-precision answer is more correct than the other.

    The second run wants about twice the memory of the first. Where there is
    not enough of it the floor comes back as `None`, which leaves the engine
    held to the strict bar for that prompt rather than stopping the sweep.
    """
    import torch

    found = reference_look(model_path, id_list, torch.float32)
    if not want_floor:
        return found
    try:
        return found, reference_look(model_path, id_list, torch.float64)
    except (MemoryError, RuntimeError) as trouble:
        if "memory" not in str(trouble).lower():
            raise
        print("  %-12s the double-precision run did not fit in memory" % "")
        return found, None


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


def graph_walk(theirs, id_count):
    """Yields the traced names in the order the graph computes them."""
    for place_index in range(id_count):
        for stem in STEP_ORDER:
            for name in sorted(theirs):
                if not name.endswith(".%d" % place_index):
                    continue
                head = name[: name.rindex(".")]
                if head.split(".")[-1] != stem:
                    continue
                # Deeper tensors inherit the error of everything before them,
                # so the bar loosens with depth rather than staying flat.
                depth = 1.0 + (head.count(".") and int(head.split(".")[1]) or 0)
                yield name, depth


def first_drift(mine, theirs, id_count, slack):
    """The first tensor in graph order that the two sides do not share."""
    for name, depth in graph_walk(theirs, id_count):
        if name not in mine:
            continue
        gap, peak = drift_of(mine[name], theirs[name])
        if gap > slack * depth * max(1.0, peak):
            return name, gap
    return None


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


def diff_real(model_path, ref_path, slack, length_list):
    """Diffs a checkpoint whose activations are rounded onto a static grid.

    Such a checkpoint does not have one answer in single precision. Every
    projection rounds what it reads and what it writes, so a sum that lands on
    a half step falls one way at one precision and the other way at another,
    and one flipped step at one layer becomes a different residual for every
    layer after it. The reference does this to itself, and no engine that does
    not reproduce its summation order bit for bit can avoid it.

    So two things are asked instead of per tensor equality. Where the graph is
    still determined the engine must be exact, and the depth it reaches is
    reported beside the depth the reference reaches against its own
    double-precision self; the embedding and the first layer, which nothing can
    have rounded twice yet, must agree outright. Where the graph is no longer
    determined, only the end of the stack is judged, against how far the
    reference moves there rather than against a constant, and over the whole
    run at once rather than prompt by prompt.
    """
    fail_count = 0
    my_peak = {"final": 0.0, "logits": 0.0}
    their_peak = {"final": 0.0, "logits": 0.0}
    size_peak = {"final": 1.0, "logits": 1.0}
    for id_count in length_list:
        mine, id_list = engine_trace(model_path, prompt_of(id_count))
        theirs, wide = reference_run(ref_path, id_list, want_floor=True)
        if not [name for name in theirs if name in mine]:
            print("  %-12s FAIL  the two sides share no tensor names" % ("%d ids" % len(id_list)))
            fail_count += 1
            continue
        my_edge = first_drift(mine, theirs, len(id_list), slack)
        their_edge = first_drift(wide, theirs, len(id_list), slack) if wide else None
        print("  %-12s first drift at %-16s %s"
              % ("%d ids" % len(id_list),
                 my_edge[0] if my_edge else "nowhere",
                 "the reference at %s" % (their_edge[0] if their_edge else "nowhere")
                 if wide else "the reference was not measured"))
        # The embedding and the first layer are computed from the weights and
        # the token alone, before anything has had a chance to be rounded twice.
        # A difference there is a defect, not a rounding choice.
        if my_edge and (my_edge[0].startswith("embed.") or my_edge[0].startswith("layer.0.")):
            print("  %-12s FAIL  %s is off by %.3e before the graph can drift"
                  % ("", my_edge[0], my_edge[1]))
            fail_count += 1
        for stem in ("final", "logits"):
            name = "%s.%d" % (stem, len(id_list) - 1)
            if name not in mine or name not in theirs:
                continue
            gap, peak = drift_of(mine[name], theirs[name])
            move = drift_of(wide[name], theirs[name])[0] if wide else 0.0
            my_peak[stem] = max(my_peak[stem], gap)
            their_peak[stem] = max(their_peak[stem], move)
            size_peak[stem] = max(size_peak[stem], peak)
            print("  %-12s      %-6s off by %.3e, the reference by %s"
                  % ("", stem, gap, "%.3e" % move if wide else "-"))
    # Judged once over the whole run rather than length by length. Whether a
    # given perturbation lands on a half step is a lottery, so a prompt where
    # the reference happens to flip nothing says nothing about the engine. What
    # the reference does across the run is the size of the movement the grid
    # allows, and the engine is held to twice it.
    for stem in ("final", "logits"):
        limit = max(slack * size_peak[stem], their_peak[stem] * FLOOR_SHARE)
        good_flag = my_peak[stem] <= limit
        fail_count += 0 if good_flag else 1
        print("  %-12s %-4s %-6s off by at most %.3e, the reference by %.3e, allowed %.3e"
              % ("over the run", "ok" if good_flag else "FAIL", stem,
                 my_peak[stem], their_peak[stem], limit))
    return fail_count


# -- the towers -----------------------------------------------------------

# The engine's dump against the reference's own tower, on whatever weights the
# checkpoint carries.
#
# Both sides start from the rows the engine says it read — the normalized
# patches, or the mel frames — because the png reader, the bicubic resize and
# the filterbank each have an independent definition in `app_test.c`, and what
# is in question here is the tower above them.


def tower_seed_rows(mine, kind):
    """The rows the engine read: normalized patches, or mel frames."""
    stem_text = "%s.%s." % (kind, "patch" if kind == "vision" else "mel")
    row_list = []
    while ("%s%d" % (stem_text, len(row_list))) in mine:
        row_list.append(list(mine["%s%d" % (stem_text, len(row_list))]))
    return row_list


def engine_media(model_path, kind, media_path):
    """Runs the engine over one picture or one clip and returns its dump."""
    handle, trace_path = tempfile.mkstemp(suffix=".igtrace")
    os.close(handle)
    room = dict(os.environ, IGLLM_TRACE=trace_path)
    line = [engine_path(), "tokens", "--model", model_path, "--prompt", "one two",
            "--image" if kind == "vision" else "--audio", media_path]
    done = subprocess.run(line, capture_output=True, text=True, env=room)
    if done.returncode != 0:
        raise RuntimeError("engine failed: %s" % done.stderr.strip())
    found = trace_read(trace_path)
    os.unlink(trace_path)
    if not found:
        raise RuntimeError("the engine wrote no activations; build with `run.py build --trace`")
    return found


def diff_tower(title, model_path, kind, media_path):
    """Holds one tower against the reference and reports what it costs.

    On a checkpoint that rounds every activation onto a static grid there is no
    single answer to agree with: a sum that lands on a half step falls one way
    here and the other way there, and one flipped step becomes a different
    residual for every layer after it. So the engine is not asked to be exact.
    It is asked to stay within the movement the grid itself allows, measured by
    nudging the reference's own input by a part in a million and seeing how far
    that carries. Anything tighter would report the checkpoint's rounding as an
    engine fault.
    """
    import app_fake
    import torch

    mine = engine_media(model_path, kind, media_path)
    seed_rows = tower_seed_rows(mine, kind)
    if not seed_rows:
        print("  %-28s FAIL  the engine read no %s rows" % (title, kind))
        return 1
    grid = None
    if kind == "vision":
        if "vision.grid.0" not in mine:
            print("  %-28s FAIL  the engine reported no patch grid" % title)
            return 1
        grid = (int(mine["vision.grid.0"][0]), int(mine["vision.grid.0"][1]))

    theirs = app_fake.tower_run(model_path, kind, seed_rows, torch.float32, grid)
    if theirs is None:
        print("  %-28s skip  the checkpoint has no %s tower" % (title, kind))
        return 0
    seed = torch.tensor(seed_rows, dtype=torch.float32)
    maker = torch.Generator().manual_seed(11)
    nudge = (seed * (1.0 + 1e-6 * torch.randn(seed.shape, generator=maker))).tolist()
    moved = app_fake.tower_run(model_path, kind, nudge, torch.float32, grid)

    row_list = []
    while ("%s.lift.%d" % (kind, len(row_list))) in mine:
        row_list.append(list(mine["%s.lift.%d" % (kind, len(row_list))]))
    if len(row_list) != theirs.shape[0]:
        print("  %-28s FAIL  the engine made %d rows, the reference %d"
              % (title, len(row_list), theirs.shape[0]))
        return 1
    ours = torch.tensor(row_list, dtype=torch.float32)
    gap = float((ours - theirs).abs().max())
    floor = float((moved - theirs).abs().max())
    peak = float(theirs.abs().max())
    limit = max(floor * FLOOR_SHARE, 1e-4 * max(1.0, peak))
    good = gap <= limit
    print("  %-28s %-4s %d rows, off by %.3e, the reference by %.3e, allowed %.3e"
          % (title, "ok" if good else "FAIL", len(row_list), gap, floor, limit))
    return 0 if good else 1


def diff_media(work_path, seed_value):
    """Builds a checkpoint with each tower and diffs both of them."""
    import app_fake

    fail_count = 0
    for kind in ("vision", "audio"):
        out_path = os.path.join(work_path, "%s-0-0" % kind)
        made = app_fake.fake_build(out_path, kind, seed_value, "real", 0, 0)
        picture_path, sound_path = made["media"]
        fail_count += diff_tower("%s tower" % kind, made["plain"], kind,
                                 picture_path if kind == "vision" else sound_path)
    return fail_count


def diff_media_at(model_path, image_path, audio_path):
    """Diffs the towers of a checkpoint that already carries them."""
    import json as json_module

    with open(os.path.join(model_path, "config.json"), "r", encoding="utf-8") as source:
        tree = json_module.load(source)
    fail_count = 0
    for kind, media_path in (("vision", image_path), ("audio", audio_path)):
        if "%s_config" % kind not in tree:
            continue
        if not media_path:
            media_path = os.path.join(model_path,
                                      "picture.png" if kind == "vision" else "sound.wav")
        if not os.path.exists(media_path):
            print("  %-28s skip  no %s to show it" % ("%s tower" % kind, kind))
            continue
        fail_count += diff_tower("%s tower" % kind, model_path, kind, media_path)
    return fail_count


# -- the seam -------------------------------------------------------------

# A tower that agrees with the reference and a text stack that agrees with the
# reference still say nothing about the join between them: the ids the processor
# lays down around a run of soft tokens, and what the model makes of them, exist
# only when the whole graph runs.
#
# Two halves, because they can be reached separately.
#
# The layout half asks the reference's own `Gemma4Processor` where the
# placeholders go, and needs no weights at all — a processor is a tokenizer and
# three small configurations — so it runs against the shipped export as easily
# as against a synthetic one.
#
# The graph half runs `Gemma4ForConditionalGeneration` over those same ids and
# compares the distribution. That needs the whole model in memory, which the
# shipped export does not fit into; a synthetic checkpoint does, and it carries
# the reference's own arrangement because the reference wrote it.
#
# The towers themselves are fed the rows the engine says it read, as `--media`
# feeds them, so that what is measured here is the seam rather than the png
# reader or the filterbank.

SEAM_PROMPT = "what is in this?"
SEAM_TOP = 8

# The cases that change which side of the join runs: no attachment at all, which
# is the frame on its own, one of each, and both together in the order a content
# list gives them.
SEAM_PLAN = (("text only", 0, 0),
             ("one image", 1, 0),
             ("one clip", 0, 1),
             ("image and clip", 1, 1))


def seam_engine(model_path, prompt, image_path, audio_path):
    """Runs the engine over one prompt and its attachments."""
    import json

    handle, trace_path = tempfile.mkstemp(suffix=".igtrace")
    os.close(handle)
    room = dict(os.environ, IGLLM_TRACE=trace_path)
    line = [engine_path(), "logits", "--model", model_path, "--prompt", prompt,
            "--serve", str(SEAM_TOP)]
    if image_path:
        line += ["--image", image_path]
    if audio_path:
        line += ["--audio", audio_path]
    done = subprocess.run(line, capture_output=True, text=True, env=room)
    if done.returncode != 0:
        raise RuntimeError("engine failed: %s" % done.stderr.strip())
    found = trace_read(trace_path)
    # The trace is not read and dropped here, because the graph half of a case
    # runs in a process of its own and reads the same file rather than being
    # handed two thousand rows down a pipe. Unlinking is the caller's.
    return json.loads(done.stdout), found, trace_path


def seam_reference_ids(processor, prompt, image_rows, audio_rows, mel_count):
    """The ids the reference lays down for the same turn.

    The counts are the engine's, because how long a run is and what surrounds it
    are separate questions and only the second is asked here; the first is asked
    by `seam_reference_count` against the processor's own arithmetic.

    Everything else is the reference's: the chat template frames the turn, and
    `Gemma4Processor` decides that a run is opened and closed and substitutes it
    for the marker the template wrote.
    """
    import numpy

    part_list = []
    if image_rows > 0:
        part_list.append({"type": "image"})
    if audio_rows > 0:
        part_list.append({"type": "audio"})
    part_list.append({"type": "text", "text": prompt})
    text = processor.tokenizer.apply_chat_template([{"role": "user", "content": part_list}],
                                                   tokenize=False, add_generation_prompt=True)
    image_part = ([processor.replace_image_token({"num_soft_tokens_per_image": [image_rows]}, 0)]
                  if image_rows > 0 else [])
    # The reference reads the run's length off the feature mask, walking it
    # through the two halvings of the subsampler, so it is handed the frames the
    # engine says it made rather than a number.
    audio_part = ([processor.replace_audio_token(
        {"input_features_mask": [numpy.ones(mel_count, dtype=bool)]}, 0)]
        if audio_rows > 0 else [])
    full_list, _ = processor.get_text_with_replacements([text], image_part, [], audio_part)
    # The template writes the opening marker itself, as the shipped one does, so
    # the tokenizer must not add a second one.
    return processor.tokenizer(full_list[0], add_special_tokens=False)["input_ids"]


def seam_reference_count(processor, image_path, audio_path):
    """How many soft tokens the processor would ask for, from the files alone.

    This is the other half of the count: not what the engine produced, but what
    the reference's own arithmetic says a picture of that size and a clip of that
    length are worth. For a picture it is the aspect-preserving resize against
    the patch budget; for a clip it is the mel framing and the subsampler.
    """
    import wave

    # `_get_num_multimodal_tokens` is the processor's own answer to this, the one
    # a serving stack asks before it allocates: the aspect-preserving resize for
    # a picture, and the mel framing and subsampler for a clip.
    size_list = None
    length_list = None
    if image_path:
        from PIL import Image

        with Image.open(image_path) as picture:
            size_list = [[picture.size[1], picture.size[0]]]
    if audio_path:
        with wave.open(audio_path, "rb") as clip:
            rate_value = getattr(processor.feature_extractor, "sampling_rate", 16000)
            length_list = [int(round(clip.getnframes() * rate_value /
                                     float(clip.getframerate())))]
    found = processor._get_num_multimodal_tokens(image_sizes=size_list, audio_lengths=length_list)
    return (found.num_image_tokens[0] if size_list else None,
            found.num_audio_tokens[0] if length_list else None)


def seam_reference_run(model, id_list, mine, grid):
    """The reference's distribution over the engine's ids and the engine's rows."""
    import torch

    room = {"input_ids": torch.tensor([id_list], dtype=torch.long)}
    patch_rows = tower_seed_rows(mine, "vision")
    mel_rows = tower_seed_rows(mine, "audio")
    if patch_rows:
        wide, high = grid
        rows = torch.tensor(patch_rows, dtype=torch.float32)[None]
        # The engine records a patch after the model-side scaling to [-1, 1].
        room["pixel_values"] = rows / 2.0 + 0.5
        room["image_position_ids"] = torch.tensor(
            [[index % wide, index // wide] for index in range(wide * high)], dtype=torch.long)[None]
    if mel_rows:
        feature = torch.tensor(mel_rows, dtype=torch.float32)[None]
        room["input_features"] = feature
        room["input_features_mask"] = torch.ones(feature.shape[:2], dtype=torch.bool)
    with torch.no_grad():
        return model(**room).logits[0, -1].float(), room


def seam_reference_floor(model, room):
    """How far the same forward moves when something that should not matter does.

    The engine is not asked to be nearer the reference than the reference is to
    itself. There are two ways to ask that, and which one applies depends on
    whether anything was attached.

    With a picture or a clip, the input itself can be nudged by a millionth, and
    what that carries through the towers is the answer. That runs the towers a
    second time, which on the shipped export is more than a sixteen gigabyte
    host can hold beside the first; where it does not fit, the reordering below
    is used instead, which does not.

    With nothing attached there is no input to nudge, and it is tempting to call
    the floor zero: the ids are exact on both sides. That is wrong on a
    checkpoint that rounds every activation onto a static grid. The ids being
    equal does not make the arithmetic exact, because a sum landing on a half
    step falls one way here and the other way there. So the same tokens are fed
    again in two chunks behind the cache — the same arithmetic in another
    summation order, which is the measure `reference_floor` uses on the text
    stack — and whatever that moves by is not a fault.
    """
    import gc

    import torch

    maker = torch.Generator().manual_seed(11)
    moved = dict(room)
    nudged = False
    for name in ("pixel_values", "input_features"):
        if name in room:
            moved[name] = room[name] * (1.0 + 1e-6 * torch.randn(room[name].shape,
                                                                 generator=maker))
            nudged = True
    # A forward on a quantized checkpoint materializes an embedding table to read
    # one row of it, so the last one has to be gone before the next is built.
    gc.collect()
    if nudged:
        try:
            with torch.no_grad():
                return model(**moved).logits[0, -1].float()
        except (RuntimeError, MemoryError) as trouble:
            if "memory" not in str(trouble).lower():
                raise
            # The nudge runs the towers a second time. Reordering the sums does
            # not, so where there is no room for the first there is usually room
            # for the second, and it measures a change that matters just as
            # little.
            del moved
            gc.collect()
    id_data = room["input_ids"]
    if id_data.shape[1] < 2:
        return 0.0
    split = id_data.shape[1] - 1
    lead = {name: value for name, value in room.items() if name != "input_ids"}
    with torch.no_grad():
        state = model(input_ids=id_data[:, :split], use_cache=True, **lead).past_key_values
        return model(input_ids=id_data[:, split:], past_key_values=state,
                     use_cache=True).logits[0, -1].float()


def seam_report(title, name, good_flag, detail):
    print("  %-16s %-14s %-4s %s" % (title, name, "ok" if good_flag else "FAIL", detail))
    return 0 if good_flag else 1


# What `--seam-case` returns. Neither is 1, which is what Python returns for an
# unhandled traceback, so a child that dies is never mistaken for a child that
# ran and disagreed.
SEAM_CASE_OKAY = 0
SEAM_CASE_APART = 2


def seam_graph(model, mine, found):
    """Judges the reference's forward against the engine's for one case."""
    import gc

    import torch

    id_list = mine["tokens"]
    grid = None
    if "vision.grid.0" in found:
        grid = (int(found["vision.grid.0"][0]), int(found["vision.grid.0"][1]))
    # A forward on a quantized checkpoint materializes an embedding table to
    # read one row of it, so the whole graph costs about a gigabyte and a half
    # above the weights. On a host that cannot spare it the ids and the soft
    # token counts still stand, and only this half is dropped.
    try:
        their_value, room = seam_reference_run(model, id_list, found, grid)
    except (RuntimeError, MemoryError) as trouble:
        if "memory" not in str(trouble).lower():
            raise
        print("  %-16s %-14s skip  the graph did not fit: %s"
              % ("", "logits", str(trouble).splitlines()[0][:60]))
        return 0
    my_value = {entry["id"]: entry["logit"] for entry in mine["top"]}
    gc.collect()
    order = torch.topk(their_value, SEAM_TOP)
    their_top = order.indices.tolist()
    gap = max((abs(my_value[slot] - float(value)) for slot, value in
               zip(their_top, order.values.tolist()) if slot in my_value), default=float("inf"))
    try:
        moved = seam_reference_floor(model, room)
    except (RuntimeError, MemoryError) as trouble:
        if "memory" not in str(trouble).lower():
            raise
        print("  %-16s %-14s skip  the floor did not fit, so the gap of %.3e is unjudged"
              % ("", "logits", gap))
        return 0
    floor = float((moved - their_value).abs().max()) if not isinstance(moved, float) else 0.0
    del room, moved
    # The nudge measures what the input is worth, not what a different
    # summation order is worth, and over sixty conformer rows the second is
    # the larger of the two. So the bar has a floor under it, wide enough for
    # float32 addition in another order and still well inside what the two
    # faults this comparison found were worth: the per-layer embedding
    # reading the placeholder moved these logits by 3.4e-03.
    limit = max(floor * FLOOR_SHARE, 1e-3)
    lead_flag = bool(mine["top"]) and mine["top"][0]["id"] == their_top[0]
    share = len([slot for slot in their_top if slot in my_value]) / float(len(their_top))
    return seam_report("", "logits", lead_flag and share >= 0.75 and gap <= limit,
                       "rank one %s, %.0f%% of the top %d shared, largest gap %.3e, "
                       "the reference moves %.3e, allowed %.3e"
                       % ("agrees" if lead_flag else "differs", 100.0 * share,
                          SEAM_TOP, gap, floor, limit))


def seam_case_alone(model_path, trace_path, mine_path):
    """The graph half of one case, in the empty process the parent gave it."""
    import json

    import torch

    from transformers.models.gemma4 import Gemma4ForConditionalGeneration

    with open(mine_path, encoding="utf-8") as source:
        mine = json.load(source)
    try:
        model = Gemma4ForConditionalGeneration.from_pretrained(model_path,
                                                               dtype=torch.float32).eval()
    except (MemoryError, RuntimeError, ValueError, OSError) as trouble:
        print("  %-16s %-14s skip  the whole model did not load: %s"
              % ("", "graph", str(trouble).split("\n")[0][:70]))
        return 0
    return seam_graph(model, mine, trace_read(trace_path))


def seam_graph_apart(model_path, trace_path, mine):
    """Runs the graph half of one case in a process that starts empty.

    Reading one row of this checkpoint's embedding table materializes the whole
    of it, and a picture carries two thousand three hundred and forty rows
    through sixteen layers besides. Held across four cases in one process, that
    is more than a sixteen gigabyte host has, and which case fits then depends
    on what ran before it — which is no way to judge anything. So each case is
    measured somewhere that starts empty and hands its memory back when it
    exits. The engine's own work is not repeated: the child reads the trace the
    parent already has, so the cost is one model load per case.
    """
    import json

    handle, mine_path = tempfile.mkstemp(suffix=".igmine")
    with os.fdopen(handle, "w", encoding="utf-8") as sink:
        json.dump(mine, sink)
    line = [sys.executable, os.path.abspath(__file__), "--seam-case",
            "--model", model_path, "--trace", trace_path, "--mine", mine_path]
    try:
        done = subprocess.run(line, capture_output=True, text=True)
    finally:
        os.unlink(mine_path)
    sys.stdout.write(done.stdout)
    if done.returncode in (SEAM_CASE_OKAY, SEAM_CASE_APART):
        return 1 if done.returncode == SEAM_CASE_APART else 0
    # Any other code is a child that died rather than answering, and the two
    # reasons want opposite treatment: a host that ran out of memory is the same
    # news as the skips the child prints itself, and anything else is a fault in
    # this harness that must not be swallowed. Neither is a disagreement between
    # the engine and the reference, so neither is counted as one, but the second
    # is printed loudly enough to chase.
    tail = [line_text for line_text in done.stderr.strip().splitlines() if line_text.strip()]
    last = tail[-1] if tail else "no output"
    if any(word in done.stderr.lower() for word in ("memory", "alloc", "0xc0000005")):
        print("  %-16s %-14s skip  the case did not fit: %s" % ("", "logits", last[:60]))
        return 0
    print("  %-16s %-14s skip  the case died, which is this harness's fault, not the "
          "engine's:" % ("", "logits"))
    for line_text in tail[-4:]:
        print("  %-16s %-14s       %s" % ("", "", line_text[:88]))
    return 0


def diff_seam(model_path, image_path, audio_path, prompt, want_graph=True):
    """Holds the join between the towers and the text stack to the reference."""
    import gc

    from transformers import AutoProcessor

    try:
        processor = AutoProcessor.from_pretrained(model_path)
    except (OSError, ValueError, KeyError) as trouble:
        # A checkpoint with no processor is a text-only export, not a broken
        # one, and there is no seam in it to walk.
        print("  %-16s %-14s skip  no processor here: %s"
              % ("", "ids", str(trouble).split("\n")[0][:60]))
        return 0
    image_token = processor.image_token_id
    audio_token = processor.audio_token_id
    fail_count = 0
    for title, want_image, want_audio in SEAM_PLAN:
        if want_image and not image_path:
            continue
        if want_audio and not audio_path:
            continue
        show_image = image_path if want_image else None
        show_audio = audio_path if want_audio else None
        mine, found, trace_path = seam_engine(model_path, prompt, show_image, show_audio)
        id_list = mine["tokens"]
        image_rows = id_list.count(image_token) if image_token is not None else 0
        audio_rows = id_list.count(audio_token) if audio_token is not None else 0
        mel_count = len(tower_seed_rows(found, "audio"))
        # The trace behind one picture is half a gigabyte, and the child is about
        # to read the same file for itself. Nothing here needs it beyond these
        # three numbers, so the parent lets go of it rather than holding a second
        # copy alongside a child that is trying to fit a forward in.
        wrote_vision = bool(tower_seed_rows(found, "vision"))
        wrote_audio = bool(tower_seed_rows(found, "audio"))
        del found
        gc.collect()
        theirs = seam_reference_ids(processor, prompt, image_rows, audio_rows, mel_count)

        same_flag = list(id_list) == list(theirs)
        detail = "%d ids" % len(id_list)
        if not same_flag:
            place = next((index for index in range(min(len(id_list), len(theirs)))
                          if id_list[index] != theirs[index]), min(len(id_list), len(theirs)))
            detail = ("%d ids against %d, first apart at %d: %s against %s"
                      % (len(id_list), len(theirs), place,
                         id_list[place] if place < len(id_list) else "-",
                         theirs[place] if place < len(theirs) else "-"))
        fail_count += seam_report(title, "ids", same_flag, detail)

        if want_image or want_audio:
            image_want, audio_want = seam_reference_count(processor, show_image, show_audio)
            if image_want is not None:
                fail_count += seam_report("", "soft tokens", image_want == image_rows,
                                          "the engine made %d rows for the picture, the "
                                          "processor asks for %d" % (image_rows, image_want))
            if audio_want is not None:
                # This was reported rather than judged while the framing behind
                # it was an open question. It is not one any more: the engine's
                # frame count is the live count `input_features_mask` marks,
                # held against the extractor across twenty-two clip lengths, so
                # a disagreement here is a fault and is called one.
                fail_count += seam_report("", "soft tokens", audio_want == audio_rows,
                                          "the engine made %d rows for the clip, the "
                                          "processor asks for %d" % (audio_rows, audio_want))

        try:
            if not want_graph or not same_flag:
                continue
            # The reference is fed the rows the engine says it read, so without
            # the activation dump there is nothing to feed it and the graph half
            # is not attempted rather than run against the wrong input.
            if (image_rows and not wrote_vision) or (audio_rows and not wrote_audio):
                print("  %-16s %-14s skip  the engine wrote no rows; build with "
                      "`run.py build --trace`" % ("", "logits"))
                continue
            fail_count += seam_graph_apart(model_path, trace_path, mine)
        finally:
            os.unlink(trace_path)
    return fail_count


def diff_seam_fake(work_path, seed_value):
    """Builds a checkpoint the reference can load whole, and walks the seam."""
    import app_fake

    made = app_fake.whole_build(os.path.join(work_path, "whole"), seed_value)
    picture_path, sound_path = made["media"]
    return diff_seam(made["plain"], picture_path, sound_path, SEAM_PROMPT, want_graph=True)


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
    parser.add_argument("--media", action="store_true",
                        help="diff the vision and audio towers instead of the text stack")
    parser.add_argument("--seam", action="store_true",
                        help="diff the join between the towers and the text stack")
    parser.add_argument("--image", help="the picture to show a vision tower")
    parser.add_argument("--audio", help="the clip to play an audio tower")
    parser.add_argument("--work", default=os.path.join(WORK_PATH, "fake"))
    parser.add_argument("--slack", type=float, default=0.0,
                        help="absolute tolerance, or 0 to measure the noise floor")
    # `--seam` spawns itself once per case, so that a forward that does not fit
    # is one case reporting a skip rather than every later case inheriting a
    # tighter host. These carry a case across that gap and are not for typing.
    parser.add_argument("--seam-case", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--trace", help=argparse.SUPPRESS)
    parser.add_argument("--mine", help=argparse.SUPPRESS)
    flag = parser.parse_args()

    if flag.seam_case:
        return SEAM_CASE_APART if seam_case_alone(flag.model, flag.trace, flag.mine) \
            else SEAM_CASE_OKAY

    try:
        import torch  # noqa: F401
    except ImportError:
        print("skip: torch is not installed; run `python3 run.py install`")
        return 0
    # `--media` builds one tower of the reference at a time, which needs the
    # gemma4 modelling code but not the language model; everything else needs
    # the whole reference.
    try:
        from transformers.models import gemma4  # noqa: F401
    except ImportError:
        print("skip: this transformers has no gemma4; run `python3 run.py install`")
        return 0
    if not os.path.exists(engine_path()):
        print("skip: no engine built; run `python3 run.py build --trace`")
        return 0

    slack = flag.slack
    fail_count = 0

    if flag.media:
        if flag.model:
            fail_count += diff_media_at(flag.model, flag.image, flag.audio)
        else:
            fail_count += diff_media(flag.work, 7)
        print("\n%d checks failed" % fail_count)
        return 0 if fail_count == 0 else 1

    if flag.seam:
        if flag.model:
            image_path = flag.image or os.path.join(flag.model, "picture.png")
            audio_path = flag.audio or os.path.join(flag.model, "sound.wav")
            print("checkpoint %s" % flag.model)
            fail_count += diff_seam(flag.model,
                                    image_path if os.path.exists(image_path) else None,
                                    audio_path if os.path.exists(audio_path) else None,
                                    SEAM_PROMPT)
        else:
            fail_count += diff_seam_fake(flag.work, 7)
        print("\n%d checks failed" % fail_count)
        return 0 if fail_count == 0 else 1

    if flag.model:
        if slack <= 0:
            slack = 1e-4
        print("checkpoint %s" % flag.model)
        fail_count += diff_real(flag.model, flag.reference or flag.model, slack,
                                LENGTH_LIST if flag.sweep else (len(PROMPT_TEXT.split()),))
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
