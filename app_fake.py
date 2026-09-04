#!/usr/bin/env python3
"""app_fake.py - builds synthetic checkpoints for the parity comparison.

The shipped Gemma 4 checkpoint is large and often unreachable, which leaves the
engine's arithmetic unverified. A synthetic checkpoint closes most of that gap:
the reference library builds a small model from a seeded initializer and writes
it out itself, so the tensor names, shapes, and layouts come from upstream
rather than from anyone's reading of upstream.

    python3 app_fake.py --out build/fake --preset dense
    python3 app_fake.py --out build/fake --preset moe --bits 4 --group 32
    python3 app_fake.py --out build/fake --preset vision
    python3 app_fake.py --out some/checkpoint --towers vision --towers audio

Three folders are produced under the output path:

    plain      the float checkpoint, written by `save_pretrained`
    packed     the same weights bit packed the way a QAT checkpoint is packed
    dequant    the packed weights folded back to float

The engine reads `packed` and the reference reads `dequant`. Both see the same
numbers, so any disagreement is the engine's unpacking rather than the loss the
quantizer introduced. What this cannot check is the naming used by the shipped
checkpoint, the real tokenizer's merges, or behaviour at full scale.

The `vision` and `audio` presets add an encoder beside the text stack, and
`--towers` adds one to a checkpoint that already exists. Those weights come from
the reference's own tower modules, the same way the text stack's come from its
own model class, so `app_diff.py --media` compares against upstream rather than
against a second reading of upstream.

`tower_open` also builds one tower of a *shipped* checkpoint on its own. That
matters because the language model of the released export dequantizes to about
nineteen gigabytes and will not fit on an ordinary machine, while a tower is a
couple of hundred megabytes: the towers can be held to the real weights even
where the whole model cannot be loaded.
"""

import argparse
import json
import math
import os
import shutil
import sys

PRESET_LIST = ("dense", "moe", "wide", "twin", "vision", "audio")

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
    # The towers stay float, as the embeddings and the norms do in the shipped
    # export, and their convolution kernels are four dimensional besides.
    if "_tower." in name or "_projector." in name:
        return False
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


# -- towers ---------------------------------------------------------------

# The vision and audio encoders, as the reference builds them.
#
# The language model of the shipped checkpoint dequantizes to about nineteen
# gigabytes, which will not fit on an ordinary machine, but each tower is a
# couple of hundred megabytes and the reference gives both a `base_model_prefix`
# so that they can be built alone. That is what makes a real comparison possible
# here at all: the towers are held against upstream on the shipped weights, the
# same standard the text stack is held to, rather than against a second reading
# of the same description.
#
# The per-module bit widths in a checkpoint are written against the full model's
# names (`vision_tower...`), so they are restated below against the standalone
# module's names. The widths and the exclusions are the checkpoint's own, and
# the unpacking is the reference's.

TOWER_PLAN = {
    "vision": {
        "stem": "model.vision_tower.",
        "lift": "model.embed_vision.",
        "bits": {".": {"num_bits": 8}},
        "keep": ["patch_embedder"],
    },
    "audio": {
        "stem": "model.audio_tower.",
        "lift": "model.embed_audio.",
        "bits": {r"^(?!.*lconv1d\.linear_start).*$": {"num_bits": 2},
                 r"lconv1d\.linear_start": {"num_bits": 4}},
        "keep": ["subsample_conv_projection", "output_proj", "relative_k_proj"],
    },
}


def tower_classes(kind):
    from transformers.models.gemma4 import configuration_gemma4 as C, modeling_gemma4 as M

    if kind == "vision":
        return M.Gemma4VisionModel, C.Gemma4VisionConfig
    return M.Gemma4AudioModel, C.Gemma4AudioConfig


def tower_form(model_path, kind):
    """The tower's configuration, as the reference's own config object."""
    _, config_class = tower_classes(kind)
    with open(os.path.join(model_path, "config.json"), "r", encoding="utf-8") as source:
        whole = json.load(source)
    if "%s_config" % kind not in whole:
        return None, whole
    field = {name: value for name, value in whole["%s_config" % kind].items()
             if not name.startswith("_")}
    field.pop("architectures", None)
    return config_class(**field), whole


def tower_open(model_path, kind, want_kind=None):
    """Builds one tower on its own and loads the checkpoint's weights into it.

    The weights stay packed; the reference's own `QuantizedLinear` decodes them,
    so nothing about the unpacking or the activation rounding comes from this
    side of the comparison.
    """
    import gc

    import torch
    from safetensors.torch import load_file
    from transformers.integrations.gemma_quant import replace_with_quant_layers
    from transformers.utils.quantization_config import GemmaQuantizationConfig

    plan = TOWER_PLAN[kind]
    model_class, _ = tower_classes(kind)
    config, whole = tower_form(model_path, kind)
    if config is None:
        return None, None
    model = model_class(config)
    if "quantization_config" in whole:
        replace_with_quant_layers(
            model,
            GemmaQuantizationConfig(num_bits=whole["quantization_config"].get("num_bits", 8),
                                    module_quant_configs=plan["bits"], quantize_embeddings=False),
            plan["keep"])

    held = load_file(os.path.join(model_path, "model.safetensors"))
    piece = {name[len(plan["stem"]):]: value.clone() for name, value in held.items()
             if name.startswith(plan["stem"]) and not name.endswith("_cache_scale")}
    del held
    gc.collect()
    missing, extra = model.load_state_dict(piece, strict=False)
    missing = [name for name in missing if not name.endswith("activation_scale")]
    if missing or extra:
        raise ValueError("the %s tower did not load: missing %s, unexpected %s"
                         % (kind, missing[:4], list(extra)[:4]))
    model.eval()
    if want_kind is not None:
        model = model.to(want_kind)
    return model, config


def tower_lift(model_path, kind, config, want_kind=None):
    """The projector that follows a tower into the text hidden width."""
    import gc

    from safetensors.torch import load_file
    from transformers.models.gemma4 import configuration_gemma4 as C, modeling_gemma4 as M

    plan = TOWER_PLAN[kind]
    with open(os.path.join(model_path, "config.json"), "r", encoding="utf-8") as source:
        whole = json.load(source)
    text = C.Gemma4TextConfig(**{name: value for name, value in
                                 whole.get("text_config", whole).items()
                                 if not name.startswith("_")})
    lift = M.Gemma4MultimodalEmbedder(config, text)
    held = load_file(os.path.join(model_path, "model.safetensors"))
    piece = {name[len(plan["lift"]):]: value.clone() for name, value in held.items()
             if name.startswith(plan["lift"])}
    del held
    gc.collect()
    lift.load_state_dict(piece, strict=False)
    lift.eval()
    if want_kind is not None:
        lift = lift.to(want_kind)
    return lift


def tower_run(model_path, kind, seed_rows, want_kind=None, grid=None):
    """Runs one tower over the rows the engine says it read, and projects them.

    Starting from the engine's own patches or mel frames is deliberate: the png
    reader, the bicubic resize and the filterbank are each held against
    independent definitions in `app_test.c`, and what is in question here is the
    tower above them.
    """
    import torch

    if want_kind is None:
        want_kind = torch.float32
    model, config = tower_open(model_path, kind, want_kind)
    if model is None:
        return None
    lift = tower_lift(model_path, kind, config, want_kind)
    rows = torch.tensor(seed_rows, dtype=want_kind)[None]
    with torch.no_grad():
        if kind == "vision":
            wide, high = grid
            # the engine records a patch after the model-side scaling to [-1, 1]
            place = torch.tensor([[index % wide, index // wide] for index in range(wide * high)],
                                 dtype=torch.long)[None]
            out = model(pixel_values=rows / 2.0 + 0.5, pixel_position_ids=place)
        else:
            mask = torch.ones(rows.shape[:2], dtype=torch.bool)
            out = model(input_features=rows, attention_mask=mask)
        return lift(inputs_embeds=out.last_hidden_state).float().reshape(-1, lift.text_hidden_size)


def tower_build(out_path, kind, seed_value=7):
    """Writes a synthetic checkpoint carrying one tower, built by the reference.

    The tower's weights, names and shapes come from the reference's own modules,
    the same way the text stack's do, so a comparison against it is a comparison
    against upstream rather than against a second reading of upstream.
    """
    import torch
    from safetensors.torch import load_file, save_file
    from transformers.models.gemma4 import configuration_gemma4 as C, modeling_gemma4 as M

    plan = TOWER_PLAN[kind]
    model_class, config_class = tower_classes(kind)
    field = dict(VISION_FORM if kind == "vision" else AUDIO_FORM)
    config = config_class(**field)
    torch.manual_seed(seed_value)
    tower = model_class(config)
    form_seed(tower, seed_value, "real")

    with open(os.path.join(out_path, "config.json"), "r", encoding="utf-8") as source:
        tree = json.load(source)
    text = C.Gemma4TextConfig(**{name: value for name, value in
                                 tree.get("text_config", tree).items()
                                 if not name.startswith("_")})
    lift = M.Gemma4MultimodalEmbedder(config, text)
    form_seed(lift, seed_value + 1, "real")

    held = load_file(os.path.join(out_path, "model.safetensors"))
    piece = {name: value.clone() for name, value in held.items()}
    del held
    for name, value in tower.state_dict().items():
        piece[plan["stem"] + name] = value.clone().contiguous()
    for name, value in lift.state_dict().items():
        piece[plan["lift"] + name] = value.clone().contiguous()
    save_file(piece, os.path.join(out_path, "model.safetensors"))

    tree["%s_config" % kind] = config.to_diff_dict() if hasattr(config, "to_diff_dict") else field
    tree.setdefault("image_token_id", 5)
    tree.setdefault("audio_token_id", 6)
    if kind == "vision":
        tree["vision_soft_tokens_per_image"] = VISION_SOFT
    with open(os.path.join(out_path, "config.json"), "w", encoding="utf-8") as target:
        json.dump(tree, target, indent=1)
    if kind == "audio":
        with open(os.path.join(out_path, "preprocessor_config.json"), "w",
                  encoding="utf-8") as target:
            json.dump(SOUND_FORM, target, indent=1)
    return tower_media(out_path, kind)


# Small enough that a whole sweep runs in seconds, and shaped so that every axis
# that could be confused for another is distinct: the two grid sides differ, the
# head divides by four but not by eight, and the pooled grid is not square.
VISION_SOFT = 9
VISION_FORM = {
    "hidden_size": 24,
    "intermediate_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 3,
    "num_key_value_heads": 3,
    "head_dim": 8,
    "patch_size": 2,
    "pooling_kernel_size": 2,
    "position_embedding_size": 32,
    "rms_norm_eps": 1e-6,
    "rope_parameters": {"rope_type": "default", "rope_theta": 100.0},
}

AUDIO_FORM = {
    "hidden_size": 24,
    "num_hidden_layers": 2,
    "num_attention_heads": 3,
    "conv_kernel_size": 3,
    # The reference sizes the subsampler's projection as
    # `channels[0] // 4 * channels[1]`, which is the filter count after two
    # halvings only when the first channel count equals the mel bin count.
    "subsampling_conv_channels": [16, 4],
    "attention_chunk_size": 4,
    "attention_context_left": 3,
    "attention_context_right": 0,
    "attention_logit_cap": 5.0,
    "residual_weight": 0.5,
    "output_proj_dims": 24,
    "rms_norm_eps": 1e-6,
    "hidden_act": "silu",
}

SOUND_FORM = {
    "feature_extractor_type": "Gemma4AudioFeatureExtractor",
    "feature_size": 16,
    "sampling_rate": 8000,
    "frame_length": 16,
    "hop_length": 8,
    "fft_length": 16,
    "mel_floor": 1e-3,
    "min_frequency": 0.0,
    "max_frequency": 4000.0,
    "preemphasis": 0.0,
    "dither": 0.0,
}


def tower_media(out_path, kind):
    """Writes one picture and one clip for a tower to be shown."""
    import struct
    import zlib

    wide, high = 27, 19
    raw = bytearray()
    for y in range(high):
        raw.append(0)
        for x in range(wide):
            raw += bytes(((x * 7 + y * 3) & 0xFF, (x * 11 + y * 5) & 0xFF, (x * x + y) & 0xFF))

    def chunk(tag, body):
        return (struct.pack(">I", len(body)) + tag + body +
                struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF))

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", wide, high, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(bytes(raw), 9))
    png += chunk(b"IEND", b"")
    picture_path = os.path.join(out_path, "picture.png")
    with open(picture_path, "wb") as target:
        target.write(png)

    rate = SOUND_FORM["sampling_rate"]
    count = rate // 4
    body = b"".join(struct.pack("<h", int(9000 * math.sin(index * 0.21) *
                                          math.cos(index * 0.013)))
                    for index in range(count))
    wave = b"RIFF" + struct.pack("<I", 36 + len(body)) + b"WAVEfmt "
    wave += struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
    wave += b"data" + struct.pack("<I", len(body)) + body
    sound_path = os.path.join(out_path, "sound.wav")
    with open(sound_path, "wb") as target:
        target.write(wave)
    del kind
    return picture_path, sound_path


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
    if preset in ("vision", "audio"):
        # Added after the packing, and to every folder, because the towers stay
        # float: both sides then read the same tower weights whichever folder
        # they were pointed at.
        for path in list(made.values()):
            made["media"] = tower_build(path, preset, seed_value)
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
    parser.add_argument("--towers", action="append", choices=("vision", "audio"),
                        help="add a tower to the checkpoint already at --out and stop")
    parser.add_argument("--seed-only", action="store_true", help=argparse.SUPPRESS)
    flag = parser.parse_args()

    try:
        import torch  # noqa: F401
    except ImportError:
        print("skip: torch is not installed; run `python3 run.py install`")
        return 0

    # `--towers` adds the encoders to a checkpoint that already has a text
    # stack, which is the only part of this builder that does not need the
    # reference library.
    if flag.towers:
        for kind in flag.towers:
            for path in tower_build(flag.out, kind, flag.seed):
                print("%-8s %s" % ("media", path))
        return 0

    try:
        # The tower presets need the reference only for the text stack, but
        # every preset needs `gemma4` itself, and a `transformers` that predates
        # it is the common case rather than the odd one.
        from transformers.models import gemma4  # noqa: F401
    except ImportError:
        print("skip: this transformers has no gemma4; run `python3 run.py install`")
        return 0

    made = fake_build(flag.out, flag.preset, flag.seed, flag.magnitude, flag.bits, flag.group)
    for name in sorted(made):
        print("%-8s %s" % (name, made[name]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
