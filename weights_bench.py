#!/usr/bin/env python3
"""Small, auditable local-weight benchmark runner; no official leaderboard scores.

Only the standard library is needed for --help, --dry-run and BFCL.
Other datasets: python -m pip install datasets==4.0.0 Pillow==12.3.0
Dataset downloads are permitted; model downloads and remote dataset code are not.
"""

import argparse
import ast
from collections import deque
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.request
import uuid


ROOT = Path(__file__).resolve().parent
BFCL_REVISION = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
BFCL_BASE = ("https://raw.githubusercontent.com/ShishirPatil/gorilla/"
             + BFCL_REVISION + "/berkeley-function-call-leaderboard/bfcl_eval/data/")
SOURCES = {
    "gsm8k": ("openai/gsm8k", "main", "test", "Reasoning: GSM8K numeric exact match"),
    "mmlu-pro": ("TIGER-Lab/MMLU-Pro", "default", "test", "Reasoning: MMLU-Pro choice accuracy"),
    "mmmu": ("MMMU/MMMU", None, "validation", "Vision: MMMU multiple-choice accuracy"),
    "bfcl": ("ShishirPatil/gorilla", "simple_python", "BFCL v4",
             "Agentic proxy: BFCL v4 simple Python strict AST match"),
}
INSTALL = "python -m pip install datasets==4.0.0 Pillow==12.3.0"
NUMBER = r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?"


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     default=str).encode()).hexdigest()


def literal(text):
    if len(text) > 100_000:
        raise ValueError("literal exceeds safety limit")
    return ast.literal_eval(text)


def unfence(text):
    text = text.strip()
    match = re.fullmatch(r"```(?:python|json)?\s*\n?(.*?)\n?```", text, re.S)
    return match.group(1).strip() if match else text


def numeric_answer(text):
    matches = re.findall(r"(?:####|(?:final\s+)?answer\s*:)\s*\$?\s*(" + NUMBER
                         + r")(?=\s|[.!?]|$)", text, re.I)
    token = matches[-1] if matches else unfence(text).strip().rstrip(".")
    if not re.fullmatch(NUMBER, token):
        return None
    try:
        number = Decimal(token.replace(",", ""))
        return number if number.is_finite() else None
    except InvalidOperation:
        return None


def choice_answer(text, count):
    matches = re.findall(r"(?:final\s+)?answer\s*:\s*\(?([A-Z])\)?(?=\W|$)",
                         text, re.I)
    candidate = matches[-1].upper() if matches else unfence(text).strip("(). \n")
    return candidate if len(candidate) == 1 and candidate in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:count] else None


def function_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return function_name(node.value) + "." + node.attr
    raise ValueError("function name must be a name or dotted name")


def parse_call(text):
    """Parse data, never execute it; only one keyword-only literal call is legal."""
    text = unfence(text)
    if len(text) > 100_000:
        raise ValueError("call exceeds safety limit")
    node = ast.parse(text, mode="eval").body
    if isinstance(node, ast.List) and len(node.elts) == 1:
        node = node.elts[0]
    if not isinstance(node, ast.Call) or node.args:
        raise ValueError("expected one call with keyword arguments")
    values = {}
    for keyword in node.keywords:
        if keyword.arg is None or keyword.arg in values:
            raise ValueError("expanded or duplicate arguments are not supported")
        values[keyword.arg] = ast.literal_eval(keyword.value)
    return function_name(node.func), values


def equal_value(left, right):
    # Python's True == 1 is not a valid argument-type match.
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal_value(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(equal_value(a, b) for a, b in zip(left, right))
    return left == right


def validate_truth(truth):
    if not isinstance(truth, list) or len(truth) != 1 or not isinstance(truth[0], dict) or len(truth[0]) != 1:
        raise ValueError("BFCL simple schema requires one reference call")
    parameters = next(iter(truth[0].values()))
    if not isinstance(parameters, dict) or any(not isinstance(v, list) or not v for v in parameters.values()):
        raise ValueError("BFCL reference arguments must contain nonempty alternative lists")


def bfcl_match(text, truth):
    validate_truth(truth)
    try:
        name, arguments = parse_call(text)
    except (ValueError, SyntaxError, TypeError, RecursionError):
        return False, None
    expected_name, expected = next(iter(truth[0].items()))
    matched = name == expected_name and not (arguments.keys() - expected.keys())
    for key, alternatives in expected.items():
        if key not in arguments:
            matched = matched and "" in alternatives
        else:
            matched = matched and any(equal_value(arguments[key], item) for item in alternatives)
    return bool(matched), (name, arguments)


def json_lines(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        data = response.read(8 * 1024 * 1024 + 1)
    if len(data) > 8 * 1024 * 1024:
        raise ValueError("dataset file exceeds 8 MiB safety limit")
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()], hashlib.sha256(data).hexdigest()


def round_robin(iterators):
    queue = deque(iterators)
    while queue:
        iterator = queue.popleft()
        try:
            item = next(iterator)
        except StopIteration:
            continue
        yield item
        queue.append(iterator)


def load_rows(name, revision, provenance):
    repo, config, split, _ = SOURCES[name]
    if name == "bfcl":
        provenance["revision"] = BFCL_REVISION
        questions, question_hash = json_lines(BFCL_BASE + "BFCL_v4_simple_python.json")
        answers, answer_hash = json_lines(BFCL_BASE + "possible_answer/BFCL_v4_simple_python.json")
        provenance.update(question_sha256=question_hash, answer_sha256=answer_hash)
        indexed = {row["id"]: row["ground_truth"] for row in answers}
        if len(indexed) != len(answers) or len({row["id"] for row in questions}) != len(questions):
            raise ValueError("duplicate BFCL example IDs")
        for row in questions:
            yield config, dict(row, ground_truth=indexed[row["id"]])
        return
    try:
        import datasets
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("Missing optional dataset dependencies. Install with: " + INSTALL) from exc
    if int(datasets.__version__.split(".")[0]) < 4:
        raise RuntimeError("datasets >=4 required (remote dataset scripts disabled). Install with: " + INSTALL)
    resolved = HfApi().dataset_info(repo, revision=revision, timeout=30).sha
    if not resolved:
        raise ValueError("dataset revision could not be resolved")
    provenance["revision"] = resolved
    configs = sorted(datasets.get_dataset_config_names(repo, revision=resolved)) if name == "mmmu" else [config]
    provenance["configs"] = configs

    def stream(config_name):
        data = datasets.load_dataset(repo, config_name, split=split, revision=resolved,
                                     streaming=True)
        if name == "mmmu":
            for key in data.features:
                if re.fullmatch(r"image_\d+", key):
                    data = data.cast_column(key, datasets.Image(decode=False))
        for row in data:
            yield config_name, row

    yield from round_robin(iter(stream(c)) for c in configs)


def prepare(name, row):
    question = row.get("question")
    if name == "bfcl":
        turns = question
        if isinstance(turns, str):
            turns = json.loads(turns)
        if isinstance(turns, list) and len(turns) == 1 and isinstance(turns[0], list):
            turns = turns[0]
        if not isinstance(turns, list) or not turns or any(
                not isinstance(turn, dict) or not isinstance(turn.get("content"), str)
                for turn in turns):
            raise ValueError("unsupported BFCL question schema")
        functions = row["function"]
        if isinstance(functions, str):
            functions = json.loads(functions)
        if not isinstance(functions, list) or not functions:
            raise ValueError("BFCL function schema is missing")
        truth = row["ground_truth"]
        validate_truth(truth)
        prompt = ("Select the function and arguments for this request. Return ONLY one Python "
                  "function call using keyword arguments and literal values; do not execute it.\n"
                  "Available functions:\n" + json.dumps(functions, ensure_ascii=False)
                  + "\nRequest:\n" + "\n".join(turn["content"] for turn in turns))
        return prompt, truth, []
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be nonempty text")
    if name == "gsm8k":
        truth = numeric_answer(str(row["answer"]))
        if truth is None:
            raise ValueError("GSM8K reference has no numeric answer")
        return ("Solve the problem. You may reason briefly. End with 'Final answer: <number>'.\n"
                + question), str(truth), []
    options = row["options"]
    if isinstance(options, str):
        options = literal(options)
    if not isinstance(options, list) or not 2 <= len(options) <= 26 or any(
            not isinstance(option, str) for option in options):
        raise ValueError("options must be 2 to 26 strings")
    answer = row.get("answer")
    if answer is None:
        answer = row["answer_index"]
    if isinstance(answer, int) and not isinstance(answer, bool):
        answer = chr(65 + answer) if 0 <= answer < len(options) else None
    truth = choice_answer(str(answer), len(options))
    if truth is None:
        raise ValueError("reference answer is not a valid option")
    prompt = ("Answer the multiple-choice question. You may reason briefly. End with "
              "'Final answer: <letter>'.\n" + question + "\n"
              + "\n".join(f"{chr(65+i)}. {option}" for i, option in enumerate(options)))
    images = []
    if name == "mmmu":
        images = [(key, row[key]) for key in sorted(row)
                  if re.fullmatch(r"image_\d+", key) and row[key] is not None]
        if not images:
            raise ValueError("MMMU vision example has no images; refusing text-only fallback")
        available = {int(key.split("_")[1]) for key, _ in images}
        referenced = {int(n) for n in re.findall(r"<image\s+(\d+)>", prompt)}
        if not referenced <= available:
            raise ValueError("MMMU question/options reference a missing image")
        prompt = ("Attached images, in order: "
                  + ", ".join("<image " + key.split("_")[1] + ">" for key, _ in images)
                  + ".\n" + prompt)
    return prompt, (truth, len(options)), images


def save_images(images, directory):
    if not images:
        return []
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Missing optional image dependency. Install with: " + INSTALL) from exc
    paths = []
    for key, value in images:
        path = directory / (key + ".png")
        if isinstance(value, dict):
            if value.get("bytes") is None:
                raise ValueError("image bytes missing; refusing dataset-supplied paths/URLs")
            source = Image.open(io.BytesIO(value["bytes"]))
        elif isinstance(value, Image.Image):
            source = value
        else:
            raise ValueError("unsupported image representation")
        try:
            source.convert("RGB").save(path)
        finally:
            source.close()
        paths.append(path)
    return paths


def command(args, prompt, images):
    line = [str(args.engine), "chat", "--model", str(args.model), "--prompt", prompt,
            "--serve", str(args.max_tokens), "--heat", "0", "--seed", str(args.seed),
            "--threads", str(args.threads), "--echo-penalty", "1"]
    if args.raw:
        line.append("--raw")
    for image in images:
        line.extend(["--image", str(image)])
    return line


def infer(args, prompt, images):
    result = subprocess.run(command(args, prompt, images), stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, encoding="utf-8", errors="replace",
                            timeout=args.timeout, check=False)
    if result.returncode:
        raise RuntimeError(f"engine exit {result.returncode}: {result.stderr[-1500:]}")
    if not result.stdout.strip():
        raise RuntimeError("engine returned no generated text")
    return result.stdout


def checkpoint_info(args):
    if not args.engine.is_file() or not os.access(args.engine, os.X_OK):
        raise RuntimeError(f"Native engine missing/not executable: {args.engine}. "
                           "Build with: python run.py build --only igllm")
    if not args.model.is_dir() or not (args.model / "config.json").is_file():
        raise RuntimeError(f"Missing local checkpoint at {args.model}; provide --model. No weights downloaded.")
    shards = sorted(args.model.glob("*.safetensors"))
    if not shards:
        raise RuntimeError("No local .safetensors weights found; no weights downloaded.")
    index = args.model / "model.safetensors.index.json"
    if index.exists():
        expected = set(json.loads(index.read_text())["weight_map"].values())
        missing = expected - {path.name for path in shards}
        if missing:
            raise RuntimeError("Missing checkpoint shards: " + ", ".join(sorted(missing)))
    metadata = {path.name: digest(path) for path in [
        args.model / "config.json", args.model / "tokenizer.json", index] if path.is_file()}
    return {"engine_sha256": digest(args.engine), "checkpoint_metadata_sha256": metadata,
            "weights_not_hashed": [{"name": p.name, "bytes": p.stat().st_size,
                                    "mtime_ns": p.stat().st_mtime_ns} for p in shards]}


def evaluate(args, name, state, work):
    provenance = state["provenance"]
    rows = load_rows(name, args.revision, provenance)
    try:
        for index, (config, row) in enumerate(rows):
            if name == "mmmu" and row.get("question_type") != "multiple-choice":
                state["skipped"] += 1
                continue
            result = {"id": str(row.get("id", row.get("question_id", index))),
                      "config": config}
            state["examples"].append(result)
            start = time.monotonic()
            image_dir = work / uuid.uuid4().hex
            image_dir.mkdir()
            try:
                prompt, truth, images = prepare(name, row)
                paths = save_images(images, image_dir)
                result["input_sha256"] = fingerprint({
                    "prompt": prompt, "truth": truth, "images": [digest(p) for p in paths]})
                result["reference"] = truth
                output = infer(args, prompt, paths)
                result["output"] = output
                if name == "bfcl":
                    correct, prediction = bfcl_match(output, truth)
                elif name == "gsm8k":
                    prediction = numeric_answer(output)
                    correct = prediction is not None and prediction == Decimal(truth)
                else:
                    prediction = choice_answer(output, truth[1])
                    correct = prediction == truth[0]
                result.update(correct=correct, prediction=prediction)
            except Exception as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                result["seconds"] = round(time.monotonic() - start, 3)
                for path in image_dir.iterdir():
                    path.unlink()
                image_dir.rmdir()
            if args.limit and len(state["examples"]) >= args.limit:
                break
        if not state["examples"]:
            raise ValueError("dataset contains no eligible examples")
        state["status"] = "ERROR" if any("error" in x for x in state["examples"]) else "OK"
    except Exception as exc:
        state.update(status="ERROR", error=f"{type(exc).__name__}: {exc}")
    finally:
        rows.close()


def cell(value):
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace(
        "|", "&#124;").replace("\r", " ").replace("\n", " ").replace("`", "&#96;")


def report(args, states, provenance):
    lines = [
        "# Local-weight benchmarks", "",
        "These are **local zero-shot diagnostics, not official leaderboard scores**. "
        "A dry run measures nothing. N/A means not run or an operational failure, never zero accuracy.", "",
        f"- Generated (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"- Local checkpoint: `{cell(args.model)}`; engine: `{cell(args.engine)}`.",
        f"- Selection: {'all eligible examples' if args.limit == 0 else f'first {args.limit} eligible examples per benchmark'} "
        "in source order; MMMU cycles through alphabetically sorted subject configurations.",
        f"- Greedy decoding: temperature 0, seed {args.seed}, repetition penalty 1, "
        f"{args.max_tokens} generated-token cap, {args.threads} thread(s), "
        f"{args.timeout:g}s timeout per inference; chat framing {'disabled (--raw)' if args.raw else 'enabled'}.",
        "- Each example starts a fresh native process (checkpoint load included in runtime). "
        "No model downloads, generated-code execution, tools, browser, remote inference, or dataset scripts.",
        "- Dataset access requires network/cache and may download large shards even for a small sample. "
        "HF revisions resolve to recorded commits; reproduce with --revision COMMIT and one --benchmarks selection. "
        "BFCL uses the fixed source commit below.", "",
        "| Benchmark / local metric | Status | Correct / completed | Errors | Skipped | Accuracy |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, state in states.items():
        examples = state["examples"]
        completed = sum("correct" in x for x in examples)
        correct = sum(x.get("correct", False) for x in examples)
        errors = sum("error" in x for x in examples) + int("error" in state)
        score = f"{100 * correct / completed:.2f}%" if completed and state["status"] == "OK" else "N/A"
        lines.append(f"| {SOURCES[name][3]} | {state['status']} | {correct} / {completed} | "
                     f"{errors} | {state['skipped']} | {score} |")
    lines += [
        "", "## Coverage and scoring limitations", "",
        "- GSM8K: official test questions; local numeric exact match after `Final answer:` or `####` "
        "(or number-only output), ignoring decimal formatting and commas; no few-shot exemplars.",
        "- MMLU-Pro: test questions; exact final option letter, no official five-shot CoT prompt.",
        "- MMMU: validation **multiple-choice only**; open-ended items excluded and counted when encountered. "
        "Every non-null image slot is supplied in numbered order, including images referenced by options. "
        "Images precede text with explicit slot labels rather than native interleaving. No text-only fallback. "
        "A small sample does not cover all subjects; --limit 0 covers all eligible MC items, not open-ended ones.",
        "- BFCL v4: **simple_python only**, single-turn, single-call, keyword/literal AST matching with "
        "reference alternatives and the empty-string omission sentinel. Strings/list order/types are strict "
        "(integer/float numeric equality allowed); argument order is ignored. Positional calls, expression "
        "evaluation, and full official normalization are not supported. This is a narrow function-calling "
        "proxy, **not** an autonomous-agent or full BFCL score.",
        "- SWE-bench Verified, GAIA, WebArena and OSWorld require separate execution/browser/desktop "
        "environments and official harnesses; they are not measured here. No aggregate across these tasks.",
        "- Tiny deterministic prefixes are smoke tests, not representative estimates. Token limits can "
        "truncate reasoning; malformed model answers count wrong. Operational/data errors make the entire "
        "affected benchmark N/A and the process exit nonzero. No retries or hidden exclusion of failures.",
        "- Metadata hashes and weight sizes/mtimes identify local artifacts only approximately: weight "
        "contents are not hashed. Greedy results can still differ across hardware/compiler builds.", "",
        "## Reproduction", "", "```sh",
        "python run.py build --only igllm",
        INSTALL,
        "python weights_bench.py --model model/ --limit 5",
        "python weights_bench.py --model model/ --limit 0  # potentially very expensive",
        "python weights_bench.py --dry-run  # no imports of ML packages or downloads",
        "```", "", "## Provenance", "",
    ]
    for name, state in states.items():
        repo, config, split, _ = SOURCES[name]
        url = (BFCL_BASE if name == "bfcl" else "https://huggingface.co/datasets/" + repo)
        lines += [f"- {name}: [{cell(repo)}]({url}), config `{cell(config or 'all subjects')}`, "
                  f"split `{cell(split)}`; {cell(json.dumps(state['provenance'], sort_keys=True))}."]
        if "error" in state:
            lines.append("  - Error: " + cell(state["error"]))
    lines += ["", "Run metadata (JSON):", "", "```json",
              json.dumps(provenance, indent=2, sort_keys=True, default=str), "```",
              "", "## Per-example audit", "",
              "Input SHA-256 covers the exact prompt, reference, and exported image hashes. "
              "Outputs below are escaped, untrusted model text, not instructions.", "",
              "| Benchmark | Config / ID | Correct | Prediction / error | Seconds | Input SHA-256 |",
              "|---|---|---|---|---:|---|"]
    for name, state in states.items():
        for item in state["examples"]:
            lines.append(f"| {name} | {cell(item['config'])} / {cell(item['id'])} | "
                         f"{item.get('correct', 'N/A')} | "
                         f"{cell(item.get('error', item.get('prediction')))} | "
                         f"{item.get('seconds', 0)} | {item.get('input_sha256', 'N/A')} |")
    for name, state in states.items():
        for item in state["examples"]:
            if "output" in item:
                lines += ["", f"**{name} / {cell(item['id'])}** — reference: "
                          + cell(item.get("reference")), "",
                          "<pre>" + cell(item["output"]) + "</pre>"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "model")
    parser.add_argument("--engine", type=Path, default=ROOT / "build" / "igllm")
    parser.add_argument("--output", type=Path, default=ROOT / "BENCHMARKS.md")
    parser.add_argument("--benchmarks", nargs="+", choices=SOURCES, default=list(SOURCES))
    parser.add_argument("--limit", type=int, default=5, help="examples per benchmark; 0 = all eligible")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=120, help="per-example native inference seconds")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--revision", default="main", help="HF dataset revision; BFCL commit is fixed")
    parser.add_argument("--raw", action="store_true", help="disable native chat framing (not output filtering)")
    parser.add_argument("--dry-run", action="store_true", help="write NOT RUN report without downloads/inference")
    args = parser.parse_args(argv)
    if args.limit < 0 or args.max_tokens < 1 or args.threads < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("limit must be >=0; max-tokens, threads and timeout must be positive")
    if not 0 <= args.seed < 2**64:
        parser.error("seed must fit an unsigned 64-bit integer")
    for name in ("model", "engine", "output"):
        setattr(args, name, getattr(args, name).resolve())
    return args


def main(argv=None):
    args = parse_args(argv)
    states = {name: {"status": "NOT RUN", "examples": [], "skipped": 0,
                     "provenance": {"requested_revision": BFCL_REVISION if name == "bfcl" else args.revision}}
              for name in args.benchmarks}
    provenance = {"harness_sha256": digest(__file__), "python": sys.version,
                  "arguments": vars(args), "packages": {}}
    for package in ("datasets", "Pillow", "huggingface_hub"):
        try:
            provenance["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            provenance["packages"][package] = "not installed"
    work = None
    try:
        if not args.dry_run:
            provenance.update(checkpoint_info(args))
            # Keep all downloaded cache/scratch files in the project, never /tmp.
            cache = ROOT / "build" / "benchmark-cache"
            cache.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("HF_HOME", str(cache / "huggingface"))
            os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
            os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
            work = cache / ("images-" + uuid.uuid4().hex)
            work.mkdir()
            for name, state in states.items():
                print(f"[benchmark] {name}", file=sys.stderr, flush=True)
                evaluate(args, name, state, work)
    except (Exception, KeyboardInterrupt) as exc:
        for state in states.values():
            if state["status"] == "NOT RUN":
                state.update(status="ERROR", error=f"{type(exc).__name__}: {exc}")
    finally:
        if work is not None:
            work.rmdir()
        report(args, states, provenance)
    print(f"Wrote {args.output}")
    return int(any(state["status"] == "ERROR" for state in states.values()))


if __name__ == "__main__":
    sys.exit(main())
