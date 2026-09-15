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
- **speculative decoding** — `--guess` proposes a block out of the stream's own
  history and verifies it in one pass, for the same text in fewer sweeps
- **mixture-of-experts** — routed blocks run beside the shared expert, with
  the stacked expert weights sliced as views
- **vision and audio** — a bidirectional patch encoder at variable resolution
  with two dimensional rotary positions, and a conformer audio encoder with
  chunked local attention, each projecting into the text embedding space
- **a picture priced by the caller** — `--image-tokens` caps what one costs,
  and the encoder is close to linear in patches, so asking for fewer soft
  tokens buys back most of the time
- **and paid for once** — `--image-keep` holds a picture's rows across runs,
  `--audio-keep` a clip's, and `--keep` reuses a prompt's cache as far as it
  agrees with the next one, so a fresh question about encoded media is a
  fraction of the first
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
| `bench`    | timed prefill and decode report; `--verbose` divides a decode step into named parts |
| `tokens`   | print the token ids of the prompt             |
| `logits`   | print the next token distribution as json     |
| `probe`    | print the resolved model shape                |
| `cache`    | print the export's calibrated cache ranges    |
| `guess`    | what a block of guesses is worth, against a proposer's ceiling; `--verbose` divides a round and prices its marginal lane |

Common flags: `--model`, `--prompt`, `--text`, `--image`, `--image-tokens`,
`--image-keep`, `--audio`, `--audio-keep`, `--serve`,
`--threads`, `--window`, `--cache`, `--heat`, `--top-k`, `--top-p`,
`--echo-penalty`, `--seed`, `--guess`, `--loop`, `--keep`, `--raw`,
`--verbose`. Run `igllm --help` for the full list.

`--guess <lanes>` is speculative decoding. A proposer guesses the next few
tokens out of the stream's own history, the model checks the whole block in one
pass, and the guesses it agrees with are kept — the only thing that changes is
how many sweeps of the weights it took. The proposer carries no second model and
no training: it asks what followed the last time this stream said what it has
just said.

**Greedy and sampled decoding give different guarantees here, and the difference
matters.** Under `--heat 0` a guess is kept when it is the argmax, so the text is
**byte for byte** the text a plain greedy run produces. Under a temperature a
guess is kept with probability `p(t)` — the model's own probability of it — and
a rejected guess is replaced by a draw from the same distribution with that
guess taken out, which is speculative sampling's rejection rule for a proposer
that names one token. That draws each token from **exactly the distribution the
plain sampler draws from**, but it is not the same *sample*: a round takes one
draw when its guess is accepted and two when it is not, so the same seed gives a
different stream at a different `--guess`, and the same stream only at the same
one. Same distribution, different sample.

That makes it worth a great deal on an answer that quotes its prompt —
summarising, editing, answering about a document, repairing code that is in the
prompt — and worth nothing on free generation, where it has only what it has
written itself. It draws nothing rather than guessing badly in that case, so the
cost of asking is a twentieth. In `chat --loop` the proposer belongs to the
conversation rather than to the turn, so a follow-up question about the same
document has both the document and the answer before it:

```
igllm chat --model model --heat 0 --guess 4 --prompt "..."

an answer that quotes the prompt   26.02 tok/s plain, 47.73 with --guess 4
free generation                    27.63 tok/s plain, 26.32 with --guess 4
guess   3.08 tokens a round over 12 rounds, 100% of 27 guesses kept
```

Under a temperature the same shape holds, with the ends further apart, because
acceptance is the model's own certainty rather than a comparison against one
token. At `--heat 1` with the default top-k 64 and top-p 0.95, repeating a
passage back verbatim — where the model is nearly certain — runs 33.55 tok/s
plain, **68.32 with `--guess 4` and 79.15 with `--guess 8`**, keeping 100% of 63
guesses and 96% of 77. That is above what the greedy path reaches on its own
quoting prompt. Free generation costs about 6%, which is where greedy sits too,
so `--guess` is a flag under a temperature for the same reason it is one without.

`guess` is where that is measured rather than asserted, and it is greedy: the
bracket it prints is about the proposer, and holding all three proposers to one
token stream is what makes the rows comparable. It runs the same greedy
continuation with three proposers — one that is always right, the n-gram
proposer that ships, and one that is always wrong — and holds all three to the
plain run's token stream, so it checks the block path as much as it measures it.
The first is the ceiling of any proposer and the last is its floor. On the
shipped export a block of sixteen brackets **2.67 to 2.82x** and a block of
eight 2.5 to 2.65, and the proposer reaches 77 to 85% of that where it applies.
A block shares the weight sweep across its lanes and cannot share the
arithmetic, which is why the ceiling is not higher; `CHANGES.md` 0.9.0 has the
argument. Run `igllm guess` for this host's own numbers — the table below is
another host's.

```
block  proposer    tok/s  ms a round  committed of drawn  vs plain  stream
8      oracle      52.08      153.61       8.00     100%     2.11x  matches plain
8      n-gram      44.27       83.41       3.69      95%     1.80x  matches plain
8      null         7.37      135.66       1.00       0%     0.30x  matches plain
```

`guess --verbose` divides a round the same way `bench --verbose` divides a step,
and subtracts the narrowest block from the widest so the difference prices the
**marginal lane** part by part — which is the whole of what holds the ceiling
down, because a block shares the weight sweep across its lanes and cannot share
the arithmetic.

```
round   a block of 2 against a block of 16, both with the oracle
part                         ms at 2     ms at 16     ms a lane    share
mlp                           36.782      160.924         8.867    49.3%
final norm, head               7.658       37.061         2.100    11.7%
score, softmax, blend          4.076       31.862         1.985    11.0%
named, in all                 67.377      319.192        17.987   100.0%
```

`bench --verbose` prints where a decode step goes, largest part first, with the
bytes each part sweeps and the rate that comes to. The parts are a partition of
the step rather than a sample of it — the timer closes one as it opens the next
— so they sum to the step, and the last two lines say what the sum misses and
what the timing itself cost.

```
phases  128 decode steps, 33.97 ms a step, 770.5 MiB swept
part                      ms a step   share MiB a step    GiB/s  a step
mlp                          16.715   48.0%      475.3    27.77    70.0
final norm, head              6.512   18.7%       97.0    14.55     1.0
q k v                         2.925    8.4%       70.1    23.41    35.0
...
unnamed 0.000 ms a step: the step's own clock, less the pass's parts
timer   0.011 ms a step of the above, 428 reads at 26 ns
```

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
quarter of the cache bytes it read as floats. It is within a percent or two of
the float cache on both builds, because the cache is read blocked by row — once
per row for the whole group of heads that shares it — rather than once per head.

What it costs is accuracy: the next token is never in doubt, and greedy decoding
diverges at the first genuinely close call — around eighty characters
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

`--image-tokens <n>` says what a picture may cost. The encoder runs before the
pooling, so every patch is paid in full and a picture is close to linear in
patches: asking for fewer soft tokens shrinks the resize and takes the tower
down with it. On four AVX2 cores a 768x512 notice is a **19.4 s turn at the
checkpoint's 280 rows and 3.3 s at `--image-tokens 40`**, transcribed exactly at
both.

It costs detail, which is why it is off by default and why there is no policy
that picks a number. The two halves of the curve part company: reading a six
line notice survives down to 35 rows, recognising a scene of four objects down
to 12, and below about six rows the model answers as though no picture were
attached at all. `CHANGES.md` 0.9.10 has both curves.

```sh
python3 run.py run -- chat --model model --image-tokens 64 \
    --image photo.png --prompt "What is in this picture?"
```

`--image-keep <path>` holds the pictures' rows in a file and reuses them next
run, which is what a photograph asked about a run at a time needs — the
in-process store only reaches a second question in the same loop. A repeat run
of the notice above is **19.60 s to 10.36 s**; what is left is the text stack
prefilling the soft tokens, which this file cannot help with because those ids
are the prompt rather than the picture — `--keep`, below, is what reaches them.
With a budget beside it, two turns about one picture are **39.44 s to 5.58 s**.

The file carries the backend and an encoder version, so one written by a
different build is refused rather than believed, and the run says so and encodes
the pictures again. It holds a working set of four and not an archive, so it
does not grow; name two paths for two working sets.

```sh
python3 run.py run -- chat --model model --image-keep pics.keep \
    --image photo.png --prompt "What is in this picture?"
```

`--audio-keep <path>` is the same file for a clip's rows, and it is a second
path rather than a section of the first so that four photographs cannot evict
the clip a conversation is about. Each reader refuses the other's file.

**Expect less of it than of `--image-keep`, and here is the number.** A picture
is half encoder and half text stack; a clip is only a **quarter** encoder, with
about seventy percent going on the prefill of its own soft tokens. So a
ten-second clip over four questions is **14.5 s to 11.1, 1.31x** on this file's
fourth host, and a thirty-second one — 750 soft tokens, the budget's ceiling —
is **40.6 s to 30.4**. The audio encoder costs 13.5 ms a soft token where the
vision one costs 35.6, and the text stack charges both about 38. **For a clip
the caching that matters is `--keep`**: with it beside this file, the same
ten-second clip answers a question it has never been asked in **about 1.8 s**,
some 8.1x. One thirty-second clip is 4.6 MiB of file, so a full store of four is
about 18 MiB.

```sh
python3 run.py run -- chat --model model --audio-keep clips.keep \
    --keep turn.cache --audio talk.wav --prompt "What do you hear?"
```

**With `--keep` beside it, a repeated turn costs almost nothing.** The two files
hold the two halves of a picture — `--image-keep` the encoder's rows and
`--keep` the prompt's cache, which is where the soft tokens have already been
prefilled — so the same turn a second time is **19.89 s to 0.53 s**, byte for
byte the same answer:

```sh
python3 run.py run -- chat --model model \
    --image-keep pics.keep --keep turn.cache \
    --image photo.png --prompt "What is in this picture?"
```

**And changing one of several pictures keeps the ones in front of it.** The
cache is marked a media run at a time rather than once over the whole prompt, so
a second turn that swaps the second of two pictures still reuses the first
picture's soft tokens: **14.50 s to 10.03**, where before it kept nothing.
Dropping a picture is **7.97 s to 1.93**. A prompt longer than the export's
512-id sliding window keeps nothing whatever it shows, so two pictures at full
resolution need `--image-tokens` beside these files to fit.

**Changing the question is nearly as cheap.** The kept cache is reused as far as
it agrees with the new prompt rather than only where the whole of it is a
prefix, and a picture's soft tokens sit in front of the question, so a question
never asked before costs **0.93 s** against the 19.99 s it costs cold — around
267 of 278 ids come off the file and only the question is primed.

Both files refuse rather than guess. A picture file written by another build is
refused and the pictures are encoded again; a cache whose prompt shows a
*different* picture behind the same placeholder ids is refused by a stamp the
ids cannot supply; and a prompt long enough to have lapped the sliding window is
primed from nothing, because rows in a ring that has turned over cannot be wound
back. Each of those costs what it cost before there was a file, and none of them
changes an answer: greedy output with the flags is byte for byte greedy output
without them.

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

`--wide` asks one thing of the build host rather than of the tier: AVX-512 VNNI
arrived two generations after the rest of AVX-512, so `run.py` asks the compiler
what `-march=native` would define here and adds `-mavx512vnni` only where the
answer says the host has it. That flag is what selects the integer kernels; a
host without it builds every other AVX-512 path and produces the float path's
results to the bit.

## Files

| file          | purpose                                        |
| ------------- | ---------------------------------------------- |
| `app_core.c`  | the engine, public interface and all eleven layers |
| `app_main.c`  | the command line front end                     |
| `app_test.c`  | the unit tests                                 |
| `app_test.py` | comparison against the transformers reference  |
| `app_fake.py` | builds a synthetic checkpoint and quantizes it |
| `app_diff.py` | layer by layer comparison against the reference |
| `app_tune.py` | fine tunes the checkpoint with TRL, and repacks the result |
| `run.py`      | install, build, test, run workflows            |
| `GUIDE.md`    | a complete tour of the implementation          |
| `CHANGES.md`  | the archive: every version, its reasoning and its refusals |
| `TODO.md`     | open items only, most consequential first      |
| `AGENTS.md`   | the conventions all of the above are written to |
| `model/`      | the vendored checkpoint, in Git LFS            |

`GUIDE.md` is the place to start if you intend to read or extend the code, and
`AGENTS.md` says how the repository is kept. `CHANGES.md` opens with the
standing results — the hosts every number is quoted against, what a token and a
picture and a clip cost, and the register of ideas measured and refused.

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

### Where it stands

**A rate means nothing without the host that produced it, so every row here
carries one.** Four cores and the wide build in both:

| host | decode | prefill | that host's own sweep ceiling |
| --- | --- | --- | --- |
| AVX-512 with VNNI and GFNI, 2.1 GHz | **28.61 tok/s** | — | 67 tok/s |
| AVX-512 with VNNI, 2.8 GHz | 21.88 | **105.5** | 41 tok/s |

The ceiling is arithmetic, not ambition. A decode step reads **784.4 MiB** — the
projections, the output head, and one row of each embedding table, which is what
`probe` and `bench` report beside the 2334.8 MiB the export maps — so a token
that spent nothing at all outside the memory would take 24 ms on a host that
sweeps at 32.18 GiB/s, and 14.9 on one that sweeps at 49.80. **Nothing that
reads the weights once per token can beat that**, and a rate quoted without its
host's sweep is not a ratio.

Passing it needs more than one committed token per sweep, which is what
`--guess` is for. On four AVX2 cores with no AVX-512 — where the multi-modal
figures above were taken — the engine runs the whole graph without the integer
kernels at all. The reference implementation, on a 2017 desktop, runs at 0.16
tokens a second.

`CHANGES.md` has every step of how these numbers were reached and what was
refused on the way; `TODO.md` has what is left.

## Fine tuning

The engine has no trainer; a tune happens on the reference side and is handed
back as a checkpoint the engine reads like any other. `app_tune.py` trains a
LoRA adapter over the frozen base weights with TRL, then folds the adapter into
the float weights so the ordinary packing path can take it:

```sh
python3 app_tune.py --model model --train   # adapter into build/tune/adapter
python3 app_tune.py --model model --merge   # folded checkpoint into build/tune/merged
python3 run.py run -- chat --model build/tune/merged --prompt "Hello!"
```

The dataset is `data_tune.jsonl`, one JSON object per line with `prompt` and
`completion`. It is a minimal example to be extended: add lines, do not
restructure it. The defaults — three epochs, rank 16, the attention projections
only — are sized for a dozen lines, and the towers stay frozen. A tuned
checkpoint is validated the same way as any other change: `run.py check --model
build/tune/merged`, with the prompts the tune was meant to move added to the
comparison.
