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
- **vision and audio** — a bidirectional patch encoder at variable resolution
  with two dimensional rotary positions, and a conformer audio encoder with
  chunked local attention, each projecting into the text embedding space
- **its own decoders** — png at every depth and interlace, jpeg sequential and
  progressive at eight or twelve bits over one, three or four components, pnm,
  bmp and riff wave readers, a bicubic
  resize, and a mel filterbank, none of them borrowed

## Quickstart

```sh
git lfs install                            # the checkpoint is stored with LFS
python3 run.py build                       # build the cli and the tests
python3 run.py test                        # build, then run the unit tests
python3 run.py run -- chat --model model \
    --prompt "Explain gravity to a child."
```

The checkpoint is vendored under `model/`, so a clone needs nothing from the
network beyond itself. It is `google/gemma-4-E2B-it-qat-mobile-transformers`,
split into three shards to stay under GitHub's two gigabyte limit for a single
LFS object; `model/README.md` says what is in it and under what licence. Any
other folder in the same layout serves just as well — `--model /path/to/folder`
— and nothing is downloaded by the engine itself.

## Tasks

| task       | purpose                                       |
| ---------- | --------------------------------------------- |
| `chat`     | an instruction-tuned turn, chat framed; `--loop` for more |
| `complete` | raw continuation of the prompt text           |
| `bench`    | timed prefill and decode report               |
| `tokens`   | print the token ids of the prompt             |
| `logits`   | print the next token distribution as json     |
| `probe`    | print the resolved model shape                |
| `cache`    | print the export's calibrated cache ranges    |

Common flags: `--model`, `--prompt`, `--text`, `--image`, `--audio`, `--serve`,
`--threads`, `--window`, `--cache`, `--heat`, `--top-k`, `--top-p`,
`--echo-penalty`, `--seed`, `--loop`, `--keep`, `--raw`, `--verbose`. Run
`igllm --help` for the full list.

`chat --loop` keeps the turn open and reads more from standard input, so a
conversation carries: what the model answered stays in the session's cache and
the next turn is framed onto it rather than replacing it. A line beginning with
a slash is an instruction rather than a turn — `/image` and `/audio` put a
picture or a clip in front of the next one, `/new` starts another conversation
on the same loaded model, `/talk n` switches between them, `/list` says what
each is holding, `/drop` closes one, `/save` and `/open` write one out and read
it back, `/help` lists them all.

Several conversations at once is the point of the split between a model and a
session: the weights are mapped once and each conversation costs only its own
cache, which `--window` sizes and `/list` reports.

`--keep <path>` holds the prompt's cache in a file and reuses it next time. On
the shipped export a 687 id prompt takes 39 seconds to prime and 2.7 seconds to
read back, and the answer is the same to the byte. What is written is the ids
the session was fed and the rows of the cache that carry anything, so the file
is the size of the prompt rather than of the window — 21 MiB for that prompt,
or 5 MiB with `--cache 8`. It is reused where the file's ids begin the prompt
about to run, and replaced where they do not; a prompt with a picture in it is
matched on the rows the tower made rather than on the ids, because two pictures
lay down the same placeholder ids. It holds a prompt rather than a
conversation: it is written before the first token is sampled, so a rerun starts
where the last run started.

`/save <path>` and `/open <path>` hold the other of the two — a conversation,
written after an answer rather than before one, so the next turn is framed onto
it and the model remembers in a later process what it said in an earlier one.
The file says which of the two it is, and the other is refused: the cache alone
cannot tell a prompt from a conversation, and reading one for the other would
drop the id the sampler stopped on or lay a turn down twice. The turns the
conversation holds ride in the caller's stamp, which is what `/list` reports
after a `/open`.

`--cache 8` holds the key and value cache as bytes, on the eight bit float grid
the export calibrates static ranges for. It is off by default. On the shipped
export it takes the cache at full span from 1803.0 MiB to 450.8 MiB — a saving
larger than the checkpoint's own mapped weights — and a decode step reads a
quarter of the cache bytes it read as floats. Until 0.8.5 it was a footprint option that cost speed —
0.8.4 measured the byte cache a third behind the float cache on the tuned build
— because every head decoded the same cached row for itself and on AVX2 the
table is read with a gather. The cache is now read blocked by row instead, once
per row for the whole group of heads that shares it, and on the same host the
byte cache is within a percent or two of the float cache on both builds: 7.22
tokens a second against 7.32 tuned, 5.87 against 6.03 default.
What it costs besides is accuracy: the next token is never in doubt, and greedy
decoding diverges at the first genuinely close call — around eighty characters
in on a short prompt, and inside twenty once the context is long enough for the
sliding window to turn over.
The `cache` task prints the calibrated ranges against the peaks a prompt
actually reaches, and what the cache costs at full span either way.

`--image` takes a png, a jpeg, a pnm or a bmp, and `--audio` a riff
wave. Each is run through its tower and put in front of the prompt, bracketed by the ids the
reference's processor brackets it with, in the place a multi-modal chat template
puts it. Either flag may be given more than once, up to eight pieces in one
prompt, and they are laid down in the order the flags appear — which is the
order a content list means by order. `--text` is the same flag for words, so an
attachment can sit in the middle of a sentence rather than in front of it;
`--prompt` still means the words that come after everything else, wherever on
the line it is written. A clip is worth a soft token every forty
milliseconds up to the processor's budget of 750 of them, and one longer than
the half minute that comes to is cut to it, which is what the reference does
with one; the run says so rather than quietly answering about the first half of
a clip:

```sh
python3 run.py run -- chat --model model \
    --image photo.png --prompt "What is in this picture?"

python3 run.py run -- chat --model model \
    --image left.png --image right.png --audio clip.wav \
    --prompt "What do these two have in common?"

python3 run.py run -- chat --model model \
    --text "Compare" --image left.png \
    --text "against" --image right.png
```

## Workflows

| command                       | effect                                        |
| ----------------------------- | --------------------------------------------- |
| `python3 run.py install`      | install the python packages parity work needs |
| `python3 run.py build`        | compile `igllm` and `igllm_test`              |
| `python3 run.py test`         | build, then run the unit tests                |
| `python3 run.py check --model <folder>` | test, then compare to the reference |
| `python3 run.py parity`       | build a synthetic checkpoint, diff layer by layer |
| `python3 run.py parity --model <folder>` | diff a real checkpoint layer by layer |
| `python3 run.py parity --media` | diff the vision and audio towers    |
| `python3 run.py parity --seam` | diff the whole multi-modal graph end to end |
| `python3 run.py run -- <args>`| build, then run the cli                       |
| `python3 run.py clean`        | remove build products                         |

Add `--debug` for an unoptimized build with the address and behaviour
sanitizers, `--tuned` to allow host specific instructions, `--wide` for the
AVX-512 kernels beside the AVX2 ones (which implies `--tuned`), or `--trace` to
compile in the activation dump the parity harness reads.

## Files

| file          | purpose                                        |
| ------------- | ---------------------------------------------- |
| `app_core.c`  | the engine, public interface and all eleven layers |
| `app_main.c`  | the command line front end                     |
| `app_test.c`  | the unit tests                                 |
| `app_test.py` | comparison against the transformers reference  |
| `app_fake.py` | builds a synthetic checkpoint and quantizes it |
| `app_diff.py` | layer by layer comparison against the reference |
| `run.py`      | install, build, test, run workflows            |
| `GUIDE.md`    | a complete tour of the implementation          |
| `CHANGES.md`  | development progress and rationale             |
| `TODO.md`     | open development tasks                         |
| `model/`      | the vendored checkpoint, in Git LFS            |

`GUIDE.md` is the place to start if you intend to read or extend the code.

## Status

The engine builds clean and passes its unit tests on POSIX and Windows, on the
scalar, SSE2, AVX2 and AVX-512 backends. Text generation, mixture-of-experts
blocks, batched prefill, and the vision and audio towers are all implemented.

Numerical parity is **verified against the reference implementation on the
shipped checkpoint**, `google/gemma-4-E2B-it-qat-mobile-transformers`, and on
synthetic weights. The engine reproduces the reference's tokenization and chat
frame exactly, agrees with it on the leading token of every test prompt, and is
identical tensor for tensor through the first layers of the stack. After those
the checkpoint's own static activation grid takes over: it rounds every
activation onto a grid fine enough that a last-bit difference in a sum becomes
a whole step, so the reference does not reproduce itself either, and both sides
move by whole steps of it. `CHANGES.md` sets out what that means and why per
tensor equality is not the criterion on a checkpoint like this one.

That result is recorded in the suite. With the export beside it, `run.py test`
runs three prompts through the whole stack and holds the ids and the head of
the distribution to what the judged build produced, so a tree that has the
checkpoint but no python still fails if a layer is wired wrong. Without a
checkpoint the section says so and passes over.

The whole multi-modal graph is verified end to end. The ids are held to the
reference's own `Gemma4Processor` — every run of soft tokens, the pair of ids
around each of them, and the frame either side — which needs no weights and so
runs against the shipped export; nine cases, up to two pictures and a clip in
one prompt, in the order the flags gave them, with the words between them as
readily as in front. The distribution over those ids is
held to `Gemma4ForConditionalGeneration`, on a synthetic checkpoint that fits in
memory, where the engine agrees with the reference to a part in ten million, and
on the shipped export for the cases whose forward fits beside its own weights.

The vision and audio towers are verified the same way, against the reference's
own tower modules on the same shipped weights. Each tower is a couple of hundred
megabytes and the reference lets one be built alone, which keeps its arithmetic
isolated from everything around it, so the towers are held to upstream rather
than to a second reading of it. On the shipped checkpoint the engine is closer
to the reference than the reference is to itself under a one-part-per-million
change to its own input; on synthetic float weights, where the activation grid
is absent, the vision tower agrees to a part in ten million.

| command                                  | what it measures                     |
| ---------------------------------------- | ------------------------------------ |
| `run.py parity`                          | eleven synthetic configurations by six prompt lengths, against the noise floor measured on each |
| `run.py parity --model <folder>`         | the real checkpoint, tensor by tensor, against how far the reference moves against itself |
| `run.py check --model <folder>`          | the next token distribution, the greedy continuation, and the speed of both sides |
| `run.py parity --media`                  | both towers against the reference's own tower modules, on the same weights |
| `run.py parity --seam`                   | the join between them: the ids the reference's processor lays down around each run of soft tokens, and the distribution the whole graph reaches over them |
| `run.py parity --seam --model model`     | the same join on the shipped export: every id and every soft token count, and every distribution that fits, each case in a process of its own. `--image` and `--audio` are repeatable here too, and two pictures of different shapes are a stronger case than one twice |

On four cores of a 2017 desktop, against the reference's 0.16 tokens a second:
decode runs at about 9.5 and prefill at about 12.4, and a build tuned for the
host — `--tuned`, which selects the AVX2 path — reaches 13.2 and 21.2.

The kernels have had four passes over them, and neither build is against the
memory yet. A decode step reads 759.4 MiB — the projections, the output head,
and one row of each embedding table, which is what `probe` and `bench` now
report beside the 2334.8 MiB the export maps. At 14.2 tokens a second that is
10.6 gigabytes a second against the 24.9 the same host hands over on a bare
sequential read, so the tuned build uses 42% of the memory and the default build
30%. The scaling says it independently: from one thread to four the memory gives
1.5 times as much and decode produces 3.2 times as many tokens.

So what is left on both builds is in spending fewer instructions as much as in
reading fewer bytes, and `TODO.md` says what the candidates are. `CHANGES.md`
0.8.1 sets out how the earlier reading of this — that decode was at the wall —
came of dividing by the key and value cache instead of the weights.

The third pass is 0.8.4's, and it is measured on a different machine, so it is
not folded into the desktop's table above. The two bit decode — 366 MiB of the
727.5 a step sweeps — now spends one broadcast per sixteen codes where it spent
two, which is worth 21% of decode at one thread on that machine and 9% at four,
and does not move a single bit of any result. 0.8.4 also closes two of the
candidates `TODO.md` was carrying: the per-group gain mirror has nothing to
convert on an export whose scales are already one `F32` a row, and the byte
cache's gather is confirmed as the wrong read rather than merely suspected.

The fourth pass is 0.8.5's, on the same machine as the third, and it is a
restructuring rather than a kernel: the cache is read blocked by row instead of
by head, so a row shared by eight heads is read — and on a byte cache decoded —
once for all of them instead of once each. Three adjacent pairs a configuration,
every pair of one sign: decode 5.91 to 7.22 tokens a second on the tuned build
with `--cache 8` and 6.96 to 7.32 with floats, prefill 14.85 to 18.94 and 19.52
to 19.69; on the default build decode 5.18 to 5.87 and 5.70 to 6.03. Not one bit
of any result moves. What it takes back is most of what the byte cache cost:
`--cache 8` was 15% behind floats on this host and is now 1%.
