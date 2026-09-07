#!/usr/bin/env python3
"""Fine-tune local Gemma 4 weights on editable prompt/response JSONL.

The packed QAT export is first expanded to floating point, without static
activation rounding. This is floating-point fine-tuning, not QAT. By default
only text normalization weights are optimized; --trainable all-text is much
more expensive. The source checkpoint is never overwritten.
"""

import argparse
import json
import math
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_examples(path):
    examples = []
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except ValueError as error:
                raise ValueError(f"{path}:{number}: invalid JSON") from error
            if not isinstance(item, dict) or any(
                not isinstance(item.get(key), str) or not item[key].strip()
                for key in ("prompt", "response")
            ):
                raise ValueError(f"{path}:{number}: need nonempty prompt and response strings")
            examples.append(item)
    if not examples:
        raise ValueError("training dataset is empty")
    return examples


def check_paths(model, output):
    if model == output or model in output.parents or output in model.parents:
        raise ValueError("--output must be separate from --model, not inside or above it")
    if output.exists():
        raise ValueError("--output already exists; choose a new directory")
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") not in ("gemma4", "gemma4_text"):
        raise ValueError("this script supports Gemma 4 checkpoints only")
    index = model / "model.safetensors.index.json"
    if index.exists():
        names = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
    else:
        names = {"model.safetensors"}
    for name in names:
        shard = (model / name).resolve()
        if model not in shard.parents:
            raise ValueError(f"checkpoint shard escapes model directory: {name}")
        with shard.open("rb") as source:
            if source.read(100).startswith(b"version https://git-lfs.github.com/spec/"):
                raise ValueError(f"{name} is a Git LFS pointer; run git lfs pull")
    return config


def tokenize_examples(tokenizer, examples, max_length):
    encoded = []
    for number, item in enumerate(examples, 1):
        messages = [{"role": "user", "content": item["prompt"]}]
        prefix = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=False
        )
        full = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": item["response"]}],
            tokenize=True, add_generation_prompt=False, return_dict=False,
        )
        if full[:len(prefix)] != prefix:
            raise ValueError(f"example {number}: chat template has inconsistent assistant prefix")
        if len(full) > max_length:
            raise ValueError(
                f"example {number}: {len(full)} tokens exceeds --max-length {max_length}; "
                "shorten it or increase the limit (examples are never silently truncated)"
            )
        if len(full) <= len(prefix):
            raise ValueError(f"example {number}: no response tokens to train")
        encoded.append({
            "input_ids": full,
            "labels": [-100] * len(prefix) + full[len(prefix):],
        })
    return encoded


def expand_quantized(model, torch):
    """Restore ordinary dense layers, including architectural embedding scales."""
    from transformers.integrations.gemma_quant import QuantizedEmbedding, QuantizedLinear
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextScaledWordEmbedding

    count = 0
    # Names, rather than module references, avoid retaining every packed tensor.
    for name in [name for name, _ in model.named_modules() if name]:
        old = model.get_submodule(name)
        if isinstance(old, QuantizedLinear):
            weight = old._dequantize_weights(model.dtype).detach()
            with torch.device("meta"):
                new = torch.nn.Linear(old.in_features, old.out_features, bias=old.bias is not None)
            new.weight = torch.nn.Parameter(weight)
            if old.bias is not None:
                new.bias = torch.nn.Parameter(old.bias.detach())
        elif isinstance(old, QuantizedEmbedding):
            weight = old.weight.detach()
            with torch.device("meta"):
                new = Gemma4TextScaledWordEmbedding(
                    old.num_embeddings, old.embedding_dim, padding_idx=None,
                    embed_scale=old.scalar_embed_scale,
                )
            new.weight = torch.nn.Parameter(weight)
            new.embed_scale = torch.tensor(old.scalar_embed_scale, device=weight.device)
        else:
            continue
        model.set_submodule(name, new)
        count += 1
    if not count:
        raise ValueError("quantized checkpoint loaded without recognized Gemma quantized layers")
    for module in model.modules():
        config = getattr(module, "config", None)
        if config is not None and hasattr(config, "quantization_config"):
            delattr(config, "quantization_config")
    for attribute in ("hf_quantizer", "quantization_method"):
        if hasattr(model, attribute):
            delattr(model, attribute)
    model.is_quantized = False
    return count


def select_parameters(model, scope):
    model.requires_grad_(False)
    selected = []
    for name, parameter in model.named_parameters():
        text = name.startswith(("model.language_model.", "model.layers.", "model.norm.",
                                "model.embed_tokens", "lm_head."))
        if text and (scope == "all-text" or "norm" in name) and parameter.is_floating_point():
            parameter.requires_grad_(True)
            selected.append((name, parameter))
    if not selected:
        raise ValueError("no trainable text parameters found")
    return selected


def train(args, examples, config):
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer, Gemma4ForConditionalGeneration
    except ImportError as error:
        raise RuntimeError(
            "install torch==2.14.0 transformers==5.16.1 safetensors==0.8.0 "
            "(choose a PyTorch build for your hardware)"
        ) from error

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model), local_files_only=True, trust_remote_code=False
    )
    encoded = tokenize_examples(tokenizer, examples, args.max_length)
    loader = Gemma4ForConditionalGeneration if config["model_type"] == "gemma4" else AutoModelForCausalLM
    model = loader.from_pretrained(
        str(args.model), local_files_only=True, trust_remote_code=False,
        dtype=dtype, attn_implementation="eager",
    )
    if config.get("quantization_config"):
        print("Expanding packed weights to dense tensors; allow tens of GB of RAM.", flush=True)
        expand_quantized(model, torch)
    model.to(device)
    selected = select_parameters(model, args.trainable)
    parameters = [parameter for _, parameter in selected]
    model.config.use_cache = False
    model.config.get_text_config().use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.0)
    losses = []
    print(f"Training {sum(p.numel() for p in parameters):,} parameters on {device}", flush=True)
    for epoch in range(args.epochs):
        order = list(range(len(encoded)))
        random.shuffle(order)
        for index in order:
            batch = {key: torch.tensor([value], device=device) for key, value in encoded[index].items()}
            batch["attention_mask"] = torch.ones_like(batch["input_ids"])
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch, use_cache=False).loss
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss; no checkpoint saved")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            if norm.item() == 0:
                raise RuntimeError("zero training gradient; no checkpoint saved")
            optimizer.step()
            losses.append(loss.item())
            print(f"epoch={epoch + 1} step={len(losses)} loss={loss.item():.6f}", flush=True)
            if len(losses) >= args.max_steps:
                break
        if len(losses) >= args.max_steps:
            break

    model.eval()
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.config.get_text_config().use_cache = True
    # Reserve a new directory only after training succeeds; never overwrite weights.
    args.output.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(str(args.output), safe_serialization=True, max_shard_size="1GB")
    tokenizer.save_pretrained(str(args.output))
    for name in ("preprocessor_config.json", "processor_config.json", "chat_template.jinja"):
        source = args.model / name
        if source.exists():
            shutil.copy2(source, args.output / name)
    metadata = {
        "source": str(args.model), "dataset": str(args.dataset), "seed": args.seed,
        "method": "dense floating-point text fine-tuning (not QAT)",
        "trainable": [name for name, _ in selected],
        "steps": len(losses), "losses": losses, "learning_rate": args.learning_rate,
        "dtype": args.dtype, "device": device, "max_length": args.max_length,
        "torch": torch.__version__, "transformers": transformers.__version__,
    }
    (args.output / "training_run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved dense checkpoint to {args.output}; source weights unchanged.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "model")
    parser.add_argument("--dataset", type=Path, default=ROOT / "weights_train.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "build" / "model-finetuned")
    parser.add_argument("--trainable", choices=("norms", "all-text"), default="norms")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--dry-run", action="store_true", help="validate paths/JSONL without ML dependencies")
    args = parser.parse_args(argv)
    if min(args.epochs, args.max_steps, args.max_length) < 1:
        parser.error("epochs, max-steps, and max-length must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate must be finite and positive")
    args.model, args.dataset, args.output = (
        path.resolve() for path in (args.model, args.dataset, args.output)
    )
    try:
        config = check_paths(args.model, args.output)
        examples = read_examples(args.dataset)
        if args.dry_run:
            print(f"Validated {len(examples)} examples; source={args.model}; output={args.output}")
            print("NOT TRAINED: tokenization, model loading, and optimization were not run.")
        else:
            train(args, examples, config)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"weights_train: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
