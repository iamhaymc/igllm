#!/usr/bin/env python3
"""app_fake.py - builds synthetic checkpoints for the parity comparison.

The shipped Gemma 4 checkpoint is large and often unreachable, which leaves the
engine's arithmetic unverified. A synthetic checkpoint closes most of that gap:
the reference library builds a small model from a seeded initializer and writes
it out itself, so the tensor names, shapes, and layouts come from upstream
rather than from anyone's reading of upstream.

    python3 app_fake.py --out build/fake --preset dense
    python3 app_fake.py --out build/fake --preset moe --bits 4 --group 32

Three folders are produced under the output path:

    plain      the float checkpoint, written by `save_pretrained`
    packed     the same weights bit packed the way a QAT checkpoint is packed
    dequant    the packed weights folded back to float

The engine reads `packed` and the reference reads `dequant`. Both see the same
numbers, so any disagreement is the engine's unpacking rather than the loss the
quantizer introduced. What this cannot check is the naming used by the shipped
checkpoint, the real tokenizer's merges, or behaviour at full scale.
"""

import argparse
import json
import os
import shutil
import sys

PRESET_LIST = ("dense", "moe", "wide", "twin")

# Kept small enough that a whole sweep runs in seconds, but not so small that
# every axis collapses: the head count, the key-value head count, and the two
# head sizes are all distinct, so a transposed or shared index shows up. The
# vocabulary has to clear the 256 byte pieces plus the specials, or the
# tokenizer emits ids the embedding table does not have.
BASE_FORM = {
    "vocab_size": 320,
    "hidden_size": 32,
    "intermediate_size": 48,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "global_head_dim": 16,
    "hidden_size_per_layer_input": 8,
    "vocab_size_per_layer_input": 320,
    "sliding_window": 8,
    "max_position_embeddings": 128,
    "rms_norm_eps": 1e-6,
}


# -- shape ----------------------------------------------------------------


def form_build(preset, extra=None):
    """Returns a text configuration for one preset."""
    from transformers.models.gemma4 import Gemma4TextConfig

    field = dict(BASE_FORM)
    if preset == "moe":
        field.update(enable_moe_block=True, num_experts=4, top_k_experts=2,
                     moe_intermediate_size=24)
    elif preset == "wide":
        field.update(use_double_wide_mlp=True)
    elif preset == "twin":
        field.update(attention_k_eq_v=True, num_global_key_value_heads=1)
    if extra:
        field.update(extra)
    return Gemma4TextConfig(**field)


def form_seed(model, seed_value, magnitude):
    """Re-randomizes every parameter so no two tensors are accidentally equal.

    The library initializes each norm weight to exactly one, which hides any
    mistake in norm indexing, and leaves the whole model symmetric under a
    permutation of layers. Real weights are neither, so `real` jitters the norms
    around one and leaves the projections at the configured spread. `wild`
    inflates everything instead, which is useful for catching a term that is
    dropped rather than mis-scaled, but agreement there is weak evidence.
    """
    import torch

    # Which parameters are norms is decided by the module they belong to, not by
    # their name. Gemma 4 has norms called `post_feedforward_layernorm_2`, and a
    # name test for `norm.weight` misses them, leaves them near zero, and mutes
    # the branch they gate. A muted branch agrees with anything.
    around_one = set()
    for stem, module in model.named_modules():
        if type(module).__name__.endswith("RMSNorm") or type(module).__name__.endswith("LayerNorm"):
            for leaf, _ in module.named_parameters(recurse=False):
                around_one.add("%s.%s" % (stem, leaf) if stem else leaf)

    generator = torch.Generator().manual_seed(seed_value)
    spread = 0.02 if magnitude == "real" else 0.5
    for name, value in model.named_parameters():
        with torch.no_grad():
            if name in around_one or name.endswith("router.scale"):
                value.copy_(1.0 + spread * 5.0 *
                            torch.randn(value.shape, generator=generator))
            elif name.endswith("per_expert_scale"):
                value.copy_(1.0 + spread * torch.randn(value.shape, generator=generator))
            else:
                value.copy_(spread * torch.randn(value.shape, generator=generator))
    return model


def form_check(model):
    """Returns the parameters that stayed suspiciously close to zero.

    A synthetic model that quietly zeroes a norm makes the branch behind it
    vanish, and a vanished branch matches any implementation at all. This is the
    guard against a comparison that passes for the wrong reason.
    """
    limp_list = []
    for name, value in model.named_parameters():
        if float(value.detach().abs().max()) < 1e-3:
            limp_list.append(name)
    return limp_list


# -- tokenizer ------------------------------------------------------------


def book_build(vocab_count):
    """Builds a small sentencepiece-shaped tokenizer.

    The engine expects Gemma's arrangement: a metaspace pre-tokenizer, `<0xNN>`
    byte pieces for the fallback, and the four special ids. Both sides load this
    same file, so tokenization is shared rather than compared, and a
    disagreement can only come from the arithmetic.
    """
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    special_list = ["<pad>", "<eos>", "<bos>", "<unk>", "<start_of_turn>", "<end_of_turn>"]
    byte_list = ["<0x%02X>" % value for value in range(256)]
    book = Tokenizer(models.BPE(unk_token="<unk>", byte_fallback=True))
    book.pre_tokenizer = pre_tokenizers.Metaspace(prepend_scheme="always")
    book.decoder = decoders.Metaspace(prepend_scheme="always")
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_count,
        special_tokens=special_list + byte_list,
        initial_alphabet=[],
        show_progress=False,
    )
    book.train_from_iterator(BOOK_CORPUS, trainer=trainer)
    return book


BOOK_CORPUS = [
    "the sea is wide and the sky is high",
    "a small stone falls to the ground",
    "gravity pulls everything toward the earth",
    "hello there, how are you today?",
    "paris is the capital of france",
    "one two three four five six seven",
]


# -- quantization ---------------------------------------------------------

# The projections a QAT checkpoint packs. The embeddings, the norms, and the
# per-layer inputs stay float, matching the shipped arrangement.
PACK_LEAF = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "per_layer_input_gate", "per_layer_projection",
    "gate_up_proj", "router.proj",
)


def pack_wanted(name):
    if not name.endswith(".weight") and not name.endswith("gate_up_proj"):
        return False
    stem = name[: -len(".weight")] if name.endswith(".weight") else name
    return any(stem.endswith(leaf) for leaf in PACK_LEAF) or stem.endswith("down_proj")


def pack_split(value, bit_count, group_size):
    """Quantizes one weight to `bit_count` bits with per-group scales.

    Returns the signed codes and the scales. The group axis is the input axis,
    which is the one the engine walks, and a group that does not divide the row
    is left short rather than padded so that the ragged case is exercised.
    """
    import torch

    row_count, col_count = value.shape[-2], value.shape[-1]
    if group_size <= 0 or group_size > col_count:
        group_size = col_count
    group_count = (col_count + group_size - 1) // group_size
    flat = value.reshape(-1, row_count, col_count).float()
    code_out = torch.zeros(flat.shape, dtype=torch.int8)
    gain_out = torch.zeros(flat.shape[0], row_count, group_count, dtype=torch.float32)
    high_code = (1 << (bit_count - 1)) - 1
    low_code = -(1 << (bit_count - 1))
    for group_index in range(group_count):
        from_index = group_index * group_size
        upto_index = min(from_index + group_size, col_count)
        part = flat[:, :, from_index:upto_index]
        peak = part.abs().amax(dim=-1)
        gain = torch.where(peak > 0, peak / high_code, torch.ones_like(peak))
        code = torch.clamp(torch.round(part / gain.unsqueeze(-1)), low_code, high_code)
        code_out[:, :, from_index:upto_index] = code.to(torch.int8)
        gain_out[:, :, group_index] = gain
    shape_list = list(value.shape)
    return (code_out.reshape(shape_list),
            gain_out.reshape(shape_list[:-2] + [row_count, group_count]))


def pack_write(from_path, into_path, dequant_path, bit_count, group_size):
    """Writes the packed checkpoint and its dequantized twin.

    Upstream's `pack_to_int32` does the packing, so the bit order is theirs and
    not a guess: a dense little-endian stream where element *i* starts at bit
    *i* * `bit_count`, with no padding between elements.
    """
    import torch
    from compressed_tensors.compressors.pack_quantized import pack_to_int32
    from safetensors.torch import load_file, save_file

    for path in (into_path, dequant_path):
        os.makedirs(path, exist_ok=True)
        for leaf in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            source = os.path.join(from_path, leaf)
            if os.path.exists(source):
                shutil.copyfile(source, os.path.join(path, leaf))

    piece = load_file(os.path.join(from_path, "model.safetensors"))
    packed_piece = {}
    plain_piece = {}
    for name in sorted(piece):
        value = piece[name]
        if not pack_wanted(name) or value.ndim < 2:
            packed_piece[name] = value
            plain_piece[name] = value
            continue
        stem = name[: -len(".weight")] if name.endswith(".weight") else name
        code, gain = pack_split(value, bit_count, group_size)
        packed_piece[stem + ".weight_packed"] = pack_to_int32(code, bit_count)
        packed_piece[stem + ".weight_scale"] = gain
        packed_piece[stem + ".weight_shape"] = torch.tensor(list(value.shape),
                                                            dtype=torch.int32)
        group_count = gain.shape[-1]
        col_count = value.shape[-1]
        spread = torch.repeat_interleave(
            gain, (col_count + group_count - 1) // group_count, dim=-1)[..., :col_count]
        plain_piece[name] = (code.float() * spread).to(value.dtype)

    packed_piece = {name: value.contiguous() for name, value in packed_piece.items()}
    plain_piece = {name: value.contiguous() for name, value in plain_piece.items()}
    save_file(packed_piece, os.path.join(into_path, "model.safetensors"))
    save_file(plain_piece, os.path.join(dequant_path, "model.safetensors"))
    rule_write(os.path.join(into_path, "config.json"), bit_count, group_size)
    return len(packed_piece)


def rule_write(path, bit_count, group_size):
    """Adds the quantization block the engine reads to decide the bit width."""
    with open(path, "r", encoding="utf-8") as source:
        tree = json.load(source)
    tree["quantization_config"] = {
        "format": "pack-quantized",
        "quant_method": "compressed-tensors",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": bit_count,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "group" if group_size > 0 else "channel",
                    "group_size": group_size if group_size > 0 else None,
                },
                "input_activations": None,
            }
        },
    }
    with open(path, "w", encoding="utf-8") as target:
        json.dump(tree, target, indent=1)


# -- workflow -------------------------------------------------------------


def fake_build(out_path, preset, seed_value, magnitude, bit_count, group_size, extra=None):
    """Writes the float checkpoint and, when asked, its packed pair."""
    import torch
    from transformers.models.gemma4 import Gemma4ForCausalLM

    plain_path = os.path.join(out_path, "plain")
    form = form_build(preset, extra)
    torch.manual_seed(seed_value)
    model = Gemma4ForCausalLM(form)
    form_seed(model, seed_value, magnitude)
    limp_list = form_check(model)
    if limp_list:
        raise ValueError("degenerate synthetic weights: %s" % ", ".join(limp_list[:4]))
    model.eval()
    os.makedirs(plain_path, exist_ok=True)
    model.save_pretrained(plain_path, safe_serialization=True)

    book = book_build(form.vocab_size)
    book.save(os.path.join(plain_path, "tokenizer.json"))
    with open(os.path.join(plain_path, "tokenizer_config.json"), "w", encoding="utf-8") as target:
        json.dump({"tokenizer_class": "PreTrainedTokenizerFast",
                   "bos_token": "<bos>", "eos_token": "<eos>",
                   "pad_token": "<pad>", "unk_token": "<unk>"}, target, indent=1)

    made = {"plain": plain_path}
    if bit_count > 0:
        packed_path = os.path.join(out_path, "packed")
        dequant_path = os.path.join(out_path, "dequant")
        pack_write(plain_path, packed_path, dequant_path, bit_count, group_size)
        made["packed"] = packed_path
        made["dequant"] = dequant_path
    return made


def main():
    parser = argparse.ArgumentParser(description="synthetic checkpoint builder")
    parser.add_argument("--out", required=True, help="folder to write the checkpoints into")
    parser.add_argument("--preset", default="dense", choices=PRESET_LIST)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--magnitude", default="real", choices=("real", "wild"))
    parser.add_argument("--bits", type=int, default=0,
                        help="bit width for the packed pair, or 0 for float only")
    parser.add_argument("--group", type=int, default=0,
                        help="group size along the input axis, or 0 for one group per row")
    flag = parser.parse_args()

    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        print("skip: torch and transformers are not installed; run `python3 run.py install`")
        return 0

    made = fake_build(flag.out, flag.preset, flag.seed, flag.magnitude, flag.bits, flag.group)
    for name in sorted(made):
        print("%-8s %s" % (name, made[name]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
