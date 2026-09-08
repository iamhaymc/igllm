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

Common flags: `--model`, `--prompt`, `--text`, `--image`, `--audio`, `--serve`,
`--threads`, `--window`, `--cache`, `--heat`, `--top-k`, `--top-p`,
`--echo-penalty`, `--seed`, `--guess`, `--loop`, `--keep`, `--raw`,
`--verbose`. Run `igllm --help` for the full list.

`--guess <lanes>` is speculative decoding, and it is greedy only. A proposer
guesses the next few tokens out of the stream's own history, the model checks
the whole block in one pass, and the guesses it agrees with are kept — so the
text is byte for byte the text a plain greedy run produces, and the only thing
that changes is how many sweeps of the weights it took. The proposer carries no
second model and no training: it asks what followed the last time this stream
said what it has just said.

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

`guess` is where that is measured rather than asserted. It runs the same greedy
continuation with three proposers — one that is always right, the n-gram
proposer that ships, and one that is always wrong — and holds all three to the
plain run's token stream, so it checks the block path as much as it measures it.
The first is the ceiling of any proposer and the last is its floor. On the
shipped export the ceiling is about 2.2x at a block of eight — 2.35x since 0.9.5
put the output head on the integer path and made a block's marginal lane 21%
cheaper — and `CHANGES.md` 0.9.0 says why it is not higher still. The proposer
reaches 85% of it where it applies. The table below was taken before 0.9.5, on a
host whose numbers `CHANGES.md` names; run `igllm guess` for this host's own.

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

So what was left on both builds, at that point, was in spending fewer
instructions as much as in reading fewer bytes. Six passes later 0.8.9 read the
first half as finished; 0.8.10 measured a step part by part and found that it
is not, on the rows this export actually has. `TODO.md` carries both halves.
`CHANGES.md` 0.8.1 sets out how the earlier reading of this — that decode was at
the wall — came of dividing by the key and value cache instead of the weights.

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

The fifth pass is 0.8.8's, on a third machine, and it is not a kernel at all.
Asking the four thread question the way `TODO.md` posed it — what are the decode
paths waiting on — turned up an answer between the kernels rather than in them.
Each path measured on its own streams the bytes a token needs in 38.9 ms at four
threads, and the token took 95.1: the missing 56 ms were the fork and the join
around every projection, of which decode issues 277 a token, at 120 microseconds
apiece on that host. Both sides of the pool now spin briefly before they sleep,
which takes the fork and join to 17.4 microseconds and decode at four threads
from 10.51 tokens a second to 15.69 on the wide build, 8.37 to 11.21 on the
tuned one and 6.19 to 7.26 on the default one. A pool with more threads than the
host has cores does not spin, because there the core a spinner holds is one
another worker needs. Not a bit of any result moves.

0.8.8 also profiled a picture again, which is the other thing `TODO.md` asked
for. At the export's full patch budget the tower takes 39.1 s single threaded —
24.0 of projections, 6.5 of scoring, 5.0 of blend and 0.83 of softmax, the last
being 0.8.6's series where 0.8.4 measured 6.6 — and the 256 soft tokens it
produces cost another 40.1 s to prefill through the text stack, which is half of
what a picture costs and was in no reading of one before. Three ways of hurrying
the projections were measured and none taken, and a fourth was: the batch's
inner loop read the spread's scratch again for every one of its sixteen lanes,
which is two loads for every multiply-add on a host that issues two of each a
cycle. Four lanes now share the row's load, and prefill on an 1800 id prompt
goes from 9.28 tokens a second to 11.40 on the default build, 16.75 to 17.80 on
the tuned one and 17.75 to 18.41 on the wide one — the order being the argument,
since the default build has no fused multiply-add and so the loads are the
largest share of what it does. Every lane's sum is the float it was, bit for
bit. `CHANGES.md` says which three were refused and why.

The sixth pass is 0.8.9's, on the same machine as the fifth, and it stops
spending instructions on arithmetic the checkpoint had already done. Every code
plane in this export ships an `input_activation_scale`, and the engine has
always rounded what goes into the product onto it — so an activation reaching a
code plane is an integer between -128 and 127 times that step, and the sum the
kernel wants is that step times an exact integer of two byte-sized factors. On a
host with AVX-512 VNNI one instruction takes sixty-four of those products where
the float loop unpacked, converted and multiplied sixteen. Against a bare sweep
of 32.18 GiB/s at four threads, the two bit path goes from 12.26 GiB/s of codes
to 25.90, the four bit from 17.37 to 29.47 and the eight bit from 23.17 to
31.04: all three are now at the memory, which 0.8.9 read as the question the
kernels can answer closed — and 0.8.10 reopened, because that bench is a row of
12288 and a decode step mostly reads rows of 1536. On the shipped export at four threads, decode goes from 8.92
tokens a second to 12.57 and prefill from 15.19 to 33.96; at one thread prefill
is 5.21 to 16.86. A picture — the tower and the 256 soft tokens it lays down,
prefilled — goes from 36.3 seconds to 20.7 at four threads and 105.0 to 48.3 at
one.

This one moves the numbers, and the integer sum is the exact one where a chain
of a thousand float products is not. Held against the reference's own tower
modules on the same weights, the vision tower goes from 2.871 off to 2.049,
where the reference moves 2.945 against itself, and the audio tower from 7.739
to 7.086 against its own 7.213. The logits are 35 layers and as many roundings
onto the export's grid further on, so exactness buys no agreement there and does
not claim any: both builds pass every check against the reference on the same
four prompts, matching its leading token and following its greedy continuation,
with gaps a little wider than before and inside the same bar. `CHANGES.md` 0.8.9
gives both tables. Builds without the instruction — every `--tuned` build, and a
`--wide` build on a host that lacks VNNI — produce the export's logits byte for
byte as they did before.

The seventh pass is 0.8.10's, on the same machine as the fifth and sixth, and
it is the one that measures before it changes anything. Six passes had been
argued from byte counts, and a byte count cannot see a part of a step that
reads no bytes. `session_step` now closes one named part as it opens the next —
one clock read a boundary, so the parts join edge to edge and sum to the step
rather than sampling it — and `bench --verbose` prints them beside the bytes
each sweeps.

The first thing that fell out is that `TODO.md`'s standing question was
mis-posed. It asked where 56 ms of an 80 ms token went and guessed: the norms,
the residual adds, the rotary turn, the per-layer embedding lookup, the forks
and joins. Measured, all of that together is under 2% of a step — the norms and
the residual adds are 0.9%, the rotary turn and the cache write 0.6%. The step
was never mostly outside the kernels.

What was outside them was smaller and more specific, and three of the four
things the table pointed at were taken. The attention's scoring and blend had
never reached the pool at all: at a 4334 id prompt they were two fifths of the
step, 38.4 ms at four threads against 39.7 at one. They are two jobs now, and
the two divide along different axes deliberately — the scores by position,
which are independent, and the blend by head, because a blend is a running sum
down the span and dividing that by position would regroup its additions and
make the engine's answer depend on the host's core count. So the export's
logits are byte for byte what they were, at one, two and four threads, and the
suite pins it with a bit comparison rather than a tolerance. The gelu between
the feed-forward's halves was the other: an eighth of a decode step and a fifth
of a prefill batch, on the calling thread, invisible to every profile before
this one because it reads no weight.

The fourth was the `tanhf` over all 262144 logits, and it was taken and put
back. The same reasoning applied — an elementwise map, nothing to lose — and
the measurement refused it: serial 3.94, 3.82, 3.87 ms against 5.20, 1.33, 5.21
forked, one fork whose workers had just been joined on the output head and had
not settled. A steady 3.87 beats a mean of 3.9 that swings by four
milliseconds, and the numbers are kept in the comment where the next person to
have the idea will find them.

On the reference host at four threads, on a 288 id prompt: decode 15.5 tokens a
second to 18.4 and prefill 43.1 to 54.8; on a 2004 id prompt decode 13.4 to
15.5, and on a 4334 id one 10.1 to 13.0 with prefill 25.0 to 33.2. Not a bit of
any logit moves.

## Where a token goes now, and what the division was hiding

0.8.11 acted on that table, and the largest thing in it turned out not to be a
kernel at all. A row of a code plane ends in one gain, and fetching it was a
call into the switch over every storage type, beside a second call to a
remainder handler that had nothing to handle — with a `vzeroupper` and the
whole caller-saved vector state spilled through the middle of the row loop for
the two of them. A block of four output rows, the remainder call guarded, and
the gain read directly where the scales are `F32`: mlp 25.90 GiB/s to 27.65,
`attn out` 25.72 to 26.84, `ple feed` 15.04 to 16.89.

Looking for it turned up three more of the same shape, none of them on the
list. The logit cap and the gelu were both calling `tanhf` — 477184 calls a
step between them — where `tanh y` is `1 - 2/(e^{2y}+1)` and the exponential is
the series the softmax has carried since 0.8.6: **4.45 ms a step to 0.28 and
2.40 to 0.29**. The gelu comes out *closer* to the closed form than the call it
replaces — 3.64e-7 against 4.31e-7 over 200001 arguments — because writing it as
`x t / (t + 1)` removes the subtraction that was losing the low bits; the cap
keeps the subtraction and stays within two and a half parts in ten million of
the cap, monotone, and exact at both ends. Neither moves the reference
comparison: every logit gap it reports is what 0.8.10 recorded. And
`kern_dot_real` had a vector path for `F32` and a scalar loop for `BF16`, while
this export keeps 26.25 MiB of bf16 that every step reads in full: `ple lift`
9.63 GiB/s to 23.30.

Decode 21.8 tokens a second to **27.1** on a 374 id prompt and 23.1 to **29.3**
on a short one, prefill 59.3 to **82.1**, the step 43.97 ms to 34.79.

The last thing the table says is what it does *not* say. 0.8.9's kernel claim
was reopened on the grounds that the output head reads 12.67 GiB/s where the
mlp reads 20.14, so the kernels must be slower on short rows. They are not:
`vpdpbusd` consumes sixty-four codes whatever their width, so a two bit plane
spends the same instruction on 16 bytes that a four bit plane spends on 32, and
**GiB/s cannot be compared across bit widths.** Counted in multiply-adds the
output head is the *fastest* plane in the step — 60.6 G a second against the
mlp's 59.0 — and the two that are really behind are the two smallest,
`ple feed` at 18.0 and `ple lift` at 12.5. That is the first entry on `TODO.md`
now.

## How a block of rows closes

0.8.12 took the first half of that entry, and the answer was in the last four
instructions of a row rather than in its loop. A row of the integer path ends in
a horizontal sum of its accumulator, a zero point correction, two multiplies and
a store, and a block of four rows was closing each of its rows on its own —
four `_mm512_reduce_add_epi32`, four sums arriving in general registers, four
rows of scalar arithmetic to put them back into floats. On a 1536 column row
that close is one epilogue against twenty-four blocks of dot product and it
vanishes into them. On `per_layer_projection`, **1536 rows of 256 columns**, it
is one against four and it costs more than what it closes.

Widening the block does nothing on its own, which is why 0.8.11 measured eight
rows against four and got a wash: eight rows of four blocks is eight closes
against thirty-two dot products exactly as four rows is four against sixteen.
The ratio is fixed by the columns. What moves it is closing the rows
*together* — four accumulators folded into one vector of four sums in fourteen
instructions instead of four reductions in near forty, four of those folds
stacked into one vector of sixteen rows, and then one subtract, one load of
sixteen gains, one broadcast and one store for the whole block. The dot products
are still taken four rows at a time, so no more than four accumulators are ever
live and the register pressure that made eight a wash never arises.

The same fold closes the batched path's four lanes, which is where prefill's
share comes from. On a second host — four cores of a Xeon at 2.1 GHz rather than
the 2.8 the figures above were taken on, so these ratios are comparable and
these absolute numbers are not — `ple feed` fell **9.8%**, `q k v` 5.5 to 6.8%,
the step floor 2.4%, prefill went **87.2 tokens a second to 93.8** and decode
24.7 to 25.3. Every logit gap the reference comparison reports is 0.8.11's to
the last digit printed.

It is not the entry closed. `ple feed` is 17.5 G multiply-adds a second to 19.4
against the 50 to 61 the other four planes reach, and what is left of it is not
the kernel at all. Two planes of 384 KiB a layer, each its own fork and join, is
seventy of them a step, and an empty fork and join of the engine's own pool
measures **2.95 us** — so 0.21 ms of the phase's 1.42, and 0.82 ms of the whole
step's 277 forks. `TODO.md` carries the measurement and what the fix would have
to be.

0.8.12 also took the one hypothesis `TODO.md` had left for the output head — that
the microbenchmark's 96 MiB stays swept because its page table entries stay hot,
where the engine's is swept once with thirty-five layers in between — and tested
it. The same 96 MiB of the mapped checkpoint costs **32.4 GiB/s back to back and
43.4 with a gigabyte of the rest of the file swept in between**: no penalty, and
anonymous memory of the same size is no faster than the file. The hypothesis is
gone and the puzzle is not.
