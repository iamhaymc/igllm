#!/usr/bin/env python3
"""run.py - install, build, test, and run workflows for the igllm engine.

The engine is a single translation unit, so every workflow reduces to one
compiler invocation. No make, cmake, or third party build tool is involved.

    python3 run.py install          # optional python packages for parity work
    python3 run.py build            # build the cli and the unit tests
    python3 run.py build --tuned    # the same, with the host's own instructions
    python3 run.py build --wide     # the same again, with the AVX-512 kernels
    python3 run.py test             # build, then run the unit tests
    python3 run.py check            # test, plus the reference comparison
    python3 run.py parity           # build a fake checkpoint and diff every layer
    python3 run.py parity --media   # the same for the vision and audio towers
    python3 run.py parity --seam    # the whole multi-modal graph, end to end
    python3 run.py run -- <args>    # build, then run the cli with <args>
    python3 run.py clean            # remove build products
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
WORK_PATH = os.path.join(ROOT_PATH, "build")

MAIN_SOURCE = "app_main.c"
TEST_SOURCE = "app_test.c"
MAIN_TARGET = "igllm"
TEST_TARGET = "igllm_test"

# `pillow` and `torchvision` are the image processor's backends, which the seam
# comparison needs to open a processor at all.
PARITY_PACKS = ["torch", "torchvision", "transformers", "compressed-tensors", "numpy",
                "safetensors", "pillow"]


# -- toolchain ------------------------------------------------------------


def tool_pick():
    """Returns (kind, program) for the first usable compiler."""
    named = os.environ.get("CC")
    if named:
        kind = "msvc" if os.path.basename(named).lower().startswith("cl") else "unix"
        return kind, named
    if platform.system() == "Windows":
        for program in ("clang", "gcc", "cl"):
            found = shutil.which(program)
            if found:
                return ("msvc" if program == "cl" else "unix"), found
    for program in ("cc", "clang", "gcc"):
        found = shutil.which(program)
        if found:
            return "unix", found
    raise SystemExit("no C compiler found; set CC to one")


# The four AVX-512 subsets the kernels reach for.  `--wide` implies `--tuned`:
# a host with these has AVX2, and every kernel without a wide path of its own is
# still the AVX2 one.
# `-mprefer-vector-width=256` keeps the compiler's own vectorization at the
# narrower width while the kernels written for sixteen lanes still get them.
# Letting it widen everything costs both halves on the host this was measured
# on: decode 3.84 tok/s against 4.12 and prefill 4.15 against 4.39.
WIDE_FLAGS = ["-mavx512f", "-mavx512bw", "-mavx512dq", "-mavx512vl",
              "-mprefer-vector-width=256"]


def tool_line(kind, program, source, target, tuned, debug, trace=False, wide=False):
    """Builds the full command line for one translation unit."""
    source_path = os.path.join(ROOT_PATH, source)
    if wide:
        tuned = True
    if kind == "msvc":
        line = [program, "/nologo", "/std:c11", "/W3", source_path]
        if trace:
            line += ["/DAPP_TRACE"]
        line += ["/Od", "/Zi"] if debug else ["/O2"]
        if wide:
            line += ["/arch:AVX512"]
        elif tuned:
            line += ["/arch:AVX2"]
        line += ["/Fe:" + target, "/Fo:" + os.path.join(WORK_PATH, "")]
        return line
    line = [program, "-std=c11", "-Wall", "-Wextra", source_path, "-o", target]
    if trace:
        line += ["-DAPP_TRACE"]
    line += ["-O0", "-g", "-fsanitize=address,undefined"] if debug else ["-O3"]
    if tuned and platform.machine().lower() in ("x86_64", "amd64", "x86", "i386", "i686"):
        line += ["-mavx2", "-mfma"]
        if wide:
            line += WIDE_FLAGS
    line += ["-lm"]
    if platform.system() != "Windows":
        line += ["-lpthread"]
    return line


def target_path(name):
    leaf = name + (".exe" if platform.system() == "Windows" else "")
    return os.path.join(WORK_PATH, leaf)


def step_show(title, line):
    print("[%s] %s" % (title, " ".join(line)), flush=True)


# -- workflows ------------------------------------------------------------


def work_install(flag):
    """Installs the python packages the parity comparison needs."""
    line = [sys.executable, "-m", "pip", "install", "--upgrade"] + PARITY_PACKS
    step_show("install", line)
    return subprocess.call(line)


def work_build(flag):
    os.makedirs(WORK_PATH, exist_ok=True)
    kind, program = tool_pick()
    plan = [(MAIN_SOURCE, MAIN_TARGET), (TEST_SOURCE, TEST_TARGET)]
    if flag.only:
        plan = [item for item in plan if item[1] == flag.only]
    for source, target in plan:
        line = tool_line(kind, program, source, target_path(target), flag.tuned, flag.debug,
                         getattr(flag, "trace", False), getattr(flag, "wide", False))
        step_show("build", line)
        code = subprocess.call(line, cwd=WORK_PATH)
        if code != 0:
            return code
    return 0


def work_test(flag):
    code = work_build(flag)
    if code != 0:
        return code
    line = [target_path(TEST_TARGET)]
    step_show("test", line)
    return subprocess.call(line)


def work_check(flag):
    code = work_test(flag)
    if code != 0:
        return code
    line = [sys.executable, os.path.join(ROOT_PATH, "app_test.py")]
    if flag.model:
        line += ["--model", flag.model]
    step_show("check", line)
    return subprocess.call(line)


def work_parity(flag):
    """Builds a synthetic checkpoint and diffs the engine against the reference."""
    flag.trace = True
    code = work_build(flag)
    if code != 0:
        return code
    line = [sys.executable, os.path.join(ROOT_PATH, "app_diff.py"), "--sweep"]
    if flag.model:
        line += ["--model", flag.model]
    if flag.media or flag.seam:
        line += ["--media" if flag.media else "--seam"]
        for path_text in flag.image or []:
            line += ["--image", path_text]
        for path_text in flag.audio or []:
            line += ["--audio", path_text]
    step_show("parity", line)
    return subprocess.call(line)


def work_run(flag):
    code = work_build(flag)
    if code != 0:
        return code
    line = [target_path(MAIN_TARGET)] + flag.rest
    step_show("run", line)
    return subprocess.call(line)


def work_clean(flag):
    del flag
    if os.path.isdir(WORK_PATH):
        shutil.rmtree(WORK_PATH)
    print("[clean] removed %s" % WORK_PATH)
    return 0


WORK_TABLE = {
    "install": work_install,
    "build": work_build,
    "test": work_test,
    "check": work_check,
    "parity": work_parity,
    "run": work_run,
    "clean": work_clean,
}


def main():
    parser = argparse.ArgumentParser(description="igllm workflows")
    parser.add_argument("work", choices=sorted(WORK_TABLE), help="workflow to perform")
    parser.add_argument("--debug", action="store_true", help="unoptimized build with sanitizers")
    parser.add_argument("--tuned", action="store_true", help="allow host specific instructions")
    parser.add_argument("--wide", action="store_true",
                        help="the AVX-512 kernels beside the AVX2 ones; implies --tuned")
    parser.add_argument("--trace", action="store_true",
                        help="compile in the activation dump the layer comparison reads")
    parser.add_argument("--only", help="build a single target")
    parser.add_argument("--model", help="checkpoint folder for the check workflow")
    parser.add_argument("--media", action="store_true",
                        help="diff the vision and audio towers rather than the text stack")
    parser.add_argument("--seam", action="store_true",
                        help="diff the join between the towers and the text stack")
    parser.add_argument("--image", action="append",
                        help="a picture to show a vision tower, repeatable")
    parser.add_argument("--audio", action="append",
                        help="a clip to play an audio tower, repeatable")
    # Everything after the first bare `--` belongs to the cli, and everything
    # before it belongs to this script. argparse.REMAINDER cannot express that:
    # it swallows the script's own flags too, so `build --debug` silently built
    # a release binary.
    argv_list = sys.argv[1:]
    rest_list = []
    if "--" in argv_list:
        split_index = argv_list.index("--")
        argv_list, rest_list = argv_list[:split_index], argv_list[split_index + 1:]
    flag = parser.parse_args(argv_list)
    flag.rest = rest_list
    return WORK_TABLE[flag.work](flag)


if __name__ == "__main__":
    sys.exit(main())
