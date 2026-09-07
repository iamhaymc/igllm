# TODO

Open development tasks, most consequential first. Everything closed is in
`CHANGES.md` under the version that closed it; the ledger at the end of this
file says which version that was, so nothing here is archaeology.

`RESEARCH.md` is the other half of this list rather than a queue feeding it.
What is here is buildable now against the shipped export. What is there needs a
trained artifact, or a hypothesis tested before it can be scoped at all.

## The ceiling, and where the engine stands against it

A token that costs one sweep of the weights cannot be made to cost less than
the sweep. On the reference host — four cores of a Xeon at 2.8 GHz, AVX-512 and
VNNI, the wide build — a bare sweep gives **32.18 GiB/s** at four threads and a
decode step reads **784.4 MiB**, so a token that spent nothing at all outside
the memory would take 24 ms: **41 tokens a second, and no more.**

0.8.10 divided the step. 0.8.11 acted on the division, and the two things it
settles both move this list.

**The first entry's premise was wrong, and the entry is closed by the
correction rather than by the work it asked for.** It read a table of GiB/s,
saw the output head at 12.67 against the mlp's 20.14, and concluded that the
kernels were slower on the short rows a step actually reads. But `vpdpbusd`
consumes sixty-four codes whatever their width, so a two bit plane spends the
same instruction on 16 bytes of weight that a four bit plane spends on 32 —
**GiB/s is not comparable across bit widths at all.** Counted in multiply-adds,
the output head is the *fastest* plane in the step: 60.6 G a second against the
mlp's 59.0, `attn out` 57.7, `q k v` 50.7. It looks slow in bytes for the same
reason it is cheap in work.

What was really costing was in the entry's other half, and was not arithmetic:
a row's epilogue was making two out-of-line calls and dragging a `vzeroupper`
and the whole caller-saved vector state through the row loop. That is fixed,
along with two `tanhf` loops the list did not have on it and a bf16 dot product
that had never been vectorized. Decode at four threads is **27.1 tokens a
second** of the 41 on a 374 id prompt and 29.3 on a short one; prefill is
**82.1**. `CHANGES.md` 0.8.11 has every number and the two refusals.

**And the correction has itself been corrected.** 0.8.11's table put the output
head at 60.6 G multiply-adds a second, the best rate in the step, and every
version since has reasoned from that. 0.8.14 found that the head does not run
the integer kernel at all — the export's `lm_head.input_activation_scale` is
`0.0`, so the plane can never be on the calibrated grid and falls to the float
spread every token. On the third host it gives **33 G multiply-adds a second
against the mlp's 52**, which is the worst rate in the step and not the best.
Whether 0.8.11's 60.6 was the float path on a faster host or an arithmetic slip
is not settled and does not need to be: the head's rate is a property of a
kernel nothing else in the step uses, and it is not comparable to the rest of
the table for that reason rather than for the bit width one.

**What is actually behind is the two smallest planes, and one of them for a
reason the first entry described exactly.** `ple feed` runs at 18.0 G
multiply-adds a second and `ple lift` at 12.5, against 50 to 61 everywhere
else. That is the first entry below. And **how does anything get past 41?** —
still only by producing more than one token per sweep, which is the second.

0.8.12 took part of the first of those. A block of rows was closing each row on
its own — a horizontal reduction into a general register, then scalar
arithmetic to put it back into a float — and on a 256 column row that close
costs more than the four dot products it closes. Sixteen rows now fold into one
vector and close together. `ple feed` moved 17.5 G multiply-adds a second to
19.4, the step floor 2.4%, and prefill 7.6% where the same fold reaches the
batched path. It is not the entry closed; what is left of it is below.

**A second host, and what it is good for.** 0.8.12's figures were taken on a
different machine from the one this section's ceiling was measured on: four
cores of a Xeon at 2.1 GHz rather than 2.8, same instruction set, where the step
floor is 40.55 ms against 34.79 and a bare four-thread sweep reaches 40 to 49
GiB/s against 32.18. Its ratios track the reference host's and its absolute
numbers do not, so anything below quoting it says so. Whichever host a
measurement is taken on, take it as the minimum a phase reaches over runs
alternating between the two builds, on a quiet machine with the checkpoint in
the page cache — 0.8.12 recorded two false results before doing that, one from a
`pip install` running beside the bench and one from a page cache that had been
evicted.

## Against llama.cpp

Several entries below note whether mainline llama.cpp has the same thing. That
is not a competitive scoreboard; it is a prior. Where llama.cpp has an idea, its
value here is known to be real and the work is porting rather than research.
Where it does not, either the idea is worse than it looks or nobody has spent
the time — and for this checkpoint, several of them look better here than they
would there, because this export's calibrated grid and untied 2-bit head are
unusual.

The notes were checked against `ggml-org/llama.cpp` at commit `465e49b`,
2026-09-07, by reading the tree. "Not in mainline" means a search of that tree
found nothing, not that an exhaustive audit was done, and it says nothing about
the forks.

## Speed

- **What is left of `ple feed`, which is probably not the kernel.** 0.8.11
  counted the step in multiply-adds instead of bytes and the ranking changed
  completely. Four planes run at 50 to 61 G multiply-adds a second and two do
  not:

  | part | ms a step | multiply-adds | G mac/s |
  | --- | --- | --- | --- |
  | final norm, head | 6.649 | 402.7 M | 60.6 |
  | mlp | 16.787 | 990.9 M | 59.0 |
  | attn out | 2.299 | 132.5 M | 57.7 |
  | q k v | 2.900 | 147.0 M | 50.7 |
  | **ple feed** | 1.532 | 27.5 M | **18.0** |
  | **ple lift** | 1.100 | 13.8 M | **12.5** |

  0.8.12 answered the *close*. `per_layer_projection` is 1536 rows of **256
  columns** — four blocks of the unpack against one epilogue, where a 1536 wide
  row has twenty-four — and a block of four rows was closing each of its rows on
  its own. Sixteen rows now fold into one vector and close in a handful of
  instructions. On the second host that is `ple feed` 17.5 G multiply-adds a
  second to 19.4, and it does not close this entry: the plane is still a third
  of the rate of the four that are not behind.

  Note what 0.8.11's refusal of eight rows a block actually established, because
  it is easy to take it the wrong way twice. Widening a block does not change
  the ratio of close to work at all — eight rows of four blocks is eight closes
  against thirty-two dot products just as four rows is four against sixteen. The
  ratio only moves if the rows close together, and that is what 0.8.12 did.

  **The dispatch was the next suspect, and 0.8.13 took it.** `ple feed` is
  `per_layer_input_gate` and `per_layer_projection`, two planes of 384 KiB a
  layer, and each is its own `pool_run` — **seventy forks and joins a step** for
  1.42 ms of work, where a fork and join used to cost 2.95 us on the second host
  and 3.06 on the third. 0.8.13 moved the publish and the collection off the
  mutex onto atomics and left the lock for waking a thread that has actually
  gone to sleep: an empty fork and join is **0.60 us at four threads** against
  3.06, and `ple feed` is 1.533 ms to 1.310 with the step floor 2.9% behind it.

  What is left in the plane is the rows, not the dispatch. At 1.31 ms it is
  still short of the four planes that run at 50 to 61 G multiply-adds a second,
  and the seventy forks it issues are now 0.04 ms rather than 0.21.

  The entry's other route is still open and is worth less than it was. Fusing
  the gate, the gelu and the lift into one fork saves 35 of the 70, but they are
  sequential — the lift needs the whole gate — so it needs a barrier inside a
  job, which `pool_group` does not have and which is a real change to it. Those
  35 forks are now about 0.02 ms a step. Do not spend the barrier on them; spend
  it only if something else wants one too.

  `ple lift` is a different thing and is not touched: `per_layer_model_projection`
  is bf16, so it is a float multiply-add path — eight lanes an instruction
  against the integer path's sixty-four — and 12.5 G a second is roughly what
  that path should give. It cannot join the integer path without a calibrated
  step it does not have. The question worth asking of it is whether the export's
  other bf16 could be quantized at load, which is a footprint change as much as
  a speed one, and which belongs with the residency item under *Not speed*.

  Together the two are 2.6 ms of a 34.8 ms step, so the ceiling on this entry
  was about 5% of a token and 0.8.12 took 2.4% of it.

- **The output head, which runs the float kernel and not the integer one.**
  96 MiB and 12.2% of everything a decode step reads, 12.36 ms of a 44.74 ms
  step on the third host, and **33 G multiply-adds a second against the mlp's
  52** — because it is not on the integer path at all.

  0.8.14 settled what three versions of this entry could not. The entry used to
  read that the head is bound by neither the memory nor the instruction count;
  it removed an instruction from the head's supposed inner loop — a quarter of
  it, on every other two and four bit plane in the step — and the head did not
  move while the four bit planes moved by ten to eighteen percent. The loop
  under test was not the loop being run.

  `kern_level_ready` requires `sheet->enter_gain > 0`, the plane's
  `input_activation_scale`, because the integer path rounds the activation onto
  that step's grid and requires it to land exactly. **The export ships
  `lm_head.input_activation_scale` as `0.0`** where every other projection has a
  real one, so the head falls to `kern_row_code` and `kern_dot_code`'s float
  spread on every token, by construction. `CHANGES.md` 0.8.14 has the
  instrumentation and the confirmation — 4 × 262144 × 1536 column-products a
  step, all of them on the float path.

  So the old entry's puzzle is not a puzzle. The 22.7 GiB/s microbenchmark ran
  the integer kernel; the engine runs the float one. The GiB/s gap is
  `kern_dot_code` against `vpdpbusd`, not two bit against four. 0.8.11's
  epilogue and 0.8.12's fold are both in the integer path and neither could ever
  have moved this plane.

  **Three routes are open, in this order.**

  *The float loop the head actually runs, which has never been written for.*
  `kern_dot_code`'s two bit path spends five instructions per sixteen codes — a
  broadcast of a dword, a variable shift, a mask, a convert and an fma — where
  the integer path spends four per sixty-four. One `vbroadcasti32x4` of sixteen
  bytes serving four shifts takes it to seventeen per sixty-four rather than
  twenty, about fifteen percent of the loop. It costs a staging pass: the four
  quarters come out in the unpack's order rather than the column's, so the
  activations have to be laid down in that order, exactly as `kern_level_stage`
  already does for the integer path. Keep the two accumulators and their slot
  assignment and the sum is bit-identical. This is the smallest of the three and
  the only one that costs nothing but work.

  *Give the head a grid, and take the integer path.* This is the large one — the
  four bit planes run at 52 G multiply-adds a second against the head's 33, and
  the head is two bit, so the ceiling is higher than that ratio suggests. It is
  not exact: the export declined to calibrate this activation, so the engine
  would be choosing a step, and that is a decision about output quality rather
  than a kernel change. Measure it as one — a step chosen per token from the
  activation's own range, against the float path's logits, over the whole
  vocabulary and not just the argmax, before any timing is quoted. If the top
  token moves on ordinary prompts the answer is no.

  *Do not score 262144 rows at all.* The entry below, unchanged by any of this
  except that the plane it prunes is five times more expensive per byte than
  that entry assumed, which makes it worth more than it looked.

  What is ruled out, and should not be tried again: the row epilogue (0.8.11,
  and it is in the wrong path anyway); eight rows a block instead of four
  (0.8.11, a wash); software prefetch of the code stream (0.8.11, 12 to 20%
  *worse* on every code plane — read the numbers before having the idea);
  sixteen rows closed together (0.8.12, wrong path); the page walk (0.8.12 — the
  same 96 MiB costs the same with a gigabyte swept in between); and the affine
  take (0.8.14, wrong path). Do not reopen any of it on a GiB/s comparison
  across bit widths, which is not a comparison at all.

- **Produce more than one token per sweep of the weights.** The only idea on
  this list that can pass 41 tokens a second, because it is the only one that
  changes the ratio of tokens to sweeps rather than the cost of a sweep.
  `RESEARCH.md` idea 1 in full; the short form is a cheap proposer guessing a
  short block and the real model checking the whole block in one batched pass,
  keeping the guesses it agrees with.

  *llama.cpp has this three ways over* — a draft model
  (`common/speculative.cpp`), lookahead decoding (`examples/lookahead`), and
  n-gram/prompt lookup with no second model at all (`common/ngram-cache`,
  `ngram-map`, `ngram-mod`). So the idea is not in doubt and neither is the
  payoff; what is unknown is this engine's verification cost and this model's
  acceptance rate.

  Start with the n-gram proposer, which needs no second model and no training,
  and measure **committed tokens per millisecond** rather than acceptance rate.
  The engine already batches prefill, so the verify side is the batched path it
  has; what it does not have is all-position logits out of `session_pass`, and a
  way to undo a rejected block.

  The correctness trap is that undo, and it is worse here than in a plain
  transformer: sliding-window layers own rings that a rejected block has already
  overwritten, and layers that share another layer's cache have to be unwound
  with it. Restoring `fill_count` is not enough. Scope the transactional cache
  before the proposer, not after.

- **The tower's own attention, tiled, with the normalizer carried.** A picture
  is now mostly this. On the reference host at one thread, the same 768 by 768
  picture at the full patch budget costs 48.3 s where it cost 105.0 before
  0.8.9; of that, prefilling the 256 soft tokens through the text stack is 16.3
  s where it was 50.7, so about **32 s is the tower** where it was about 54. The
  projections took the integer path with everything else. The scoring and the
  blend did not — they are float, untouched since they were written, and a
  larger share of a tower than they have ever been.

  `RESEARCH.md` idea 11: score a tile, carry the running maximum and normalizer,
  accumulate the blend into the output, and never hold the whole score matrix.
  It is a memory schedule rather than a kernel, and at 2304 patches the score
  matrix it stops materializing is 2304 by 2304.

  *llama.cpp has this on CPU* —
  `ggml_compute_forward_flash_attn_ext_tiled` with a partial-reduction pass, in
  `ggml/src/ggml-cpu/ops.cpp`. Which is the strongest argument that it is worth
  the summation-order change it costs.

  Retake 0.8.8's profile first — projections, scoring, blend, softmax, one
  thread — because 0.8.9 changed the shares it recorded and the schedule should
  be written against the new ones.

- **Stop scoring 262144 rows to pick one.** The output head is untied on this
  export and 2-bit: 262144 by 1536 is **96.0 MiB, 12.2% of everything a decode
  step reads**, spent to find the largest of 262144 numbers and then throw the
  rest away.

  `RESEARCH.md` idea 10: cluster the head's rows once at load, bound each
  cluster cheaply, evaluate the promising ones, and discard a cluster only when
  its bound cannot beat the best candidate so far. Exact for greedy decoding,
  and exact for top-k where every excluded row is provably under the retained
  threshold. Arbitrary top-p cannot ignore the discarded tail and would fall
  back to the full head.

  *Not in mainline llama.cpp* — it always computes the full head. Two reasons
  it may be worth more here than there: this head is untied, so it is a real
  96 MiB rather than a table the embedding lookup already touched, and 2-bit
  codes make the per-row bound cheap to precompute.

  Removing the head entirely would only move the 41 tokens-a-second ceiling to
  about 47, so this is a multiplier and not a strategy. Measure **bytes actually
  not read**, not clusters skipped, and stop if the bound needs most of a row to
  be useful.

  0.8.14 changed what this is worth without changing the entry. The head runs
  the float kernel rather than the integer one, so the bytes this would stop
  reading are five times more expensive per byte than the entry assumed when it
  wrote them off as "the best rate in the step". A bound that skips half the
  head is worth half of 12.36 ms here, not half of 6.65.

  This entry used to be scoped together with the logit cap, on the argument that
  a bound which never materializes most of the head makes capping most of the
  head moot. 0.8.11 closed the cap on its own — 4.45 ms a step to 0.28 — so the
  two are no longer one piece of work, and what is left to win here is the head
  itself: 6.65 ms of a 34.79 ms step, with the cap and the sampler that used to
  ride with it now 0.28 and 0.78.

- **Fewer patches, before the pooling — and a budget the caller can ask for.**
  The largest vision win available, and the one that costs behaviour.

  The tower is 16 layers of width 768, 12 heads, 16-pixel patches, and the 3×3
  pooling that turns 2304 patches into 256 soft tokens happens **after** the
  encoder. So every patch is paid in full and the pooling saves nothing but the
  text stack's share. At one quarter the patches the projections are roughly a
  quarter and the dense attention pair work roughly a sixteenth.

  Two halves, and they are separable. The cheap half is `RESEARCH.md` idea 4: a
  patch budget the caller states, honestly plumbed, instead of the configured
  maximum every time — with the 3×3 pool geometry, the positions and the emitted
  token count all still valid. The expensive half is idea 5: merge redundant
  patches inside the encoder after the first few blocks, keeping enough
  provenance to reconstruct the pooling contract.

  *llama.cpp has the budget knob and not the merging* — `image_min_tokens` and
  `image_max_tokens` in `tools/mtmd/mtmd.h`, read from metadata and overridable
  by the caller, with no policy that escalates after a cheap pass. And no
  merging or pruning pass in `clip.cpp`: the one mention of token merging there
  describes a model variant whose own convolution does it, not a runtime that
  reduces patches. So the cheap half is a known-good shape to copy and the
  expensive half is unexplored ground on CPU.

  Both are approximate. Report a quality curve over grid sizes split by OCR
  versus coarse understanding, not a single latency number; a fast path that
  falls back to full resolution half the time is not a fast path.

- **Reuse a picture's rows when it is the same picture.** Exact, cheap, and
  narrow: cache the post-projector rows against a collision-resistant identity
  of the pixels plus the preprocessing and model configuration, so a second
  question about the same image costs nothing. `RESEARCH.md` idea 12.

  *llama.cpp has a narrower form of this.* Its server pushes a placeholder for
  an encoded media chunk into the slot's prompt tokens — "the chunk is already
  in the KV cache at this point, so we don't need to keep its data around"
  (`tools/server/server-context.cpp`) — so a picture is reused by an ordinary
  **prompt-prefix match within one slot**, keyed by the caller's bitmap id. That
  covers the follow-up turn and nothing else: not the same picture in another
  conversation, not with different text in front of it, not after a restart.
  There is no content-addressed store of post-projector rows anywhere in the
  tree. The gap is worth having, and the prefix form is worth having too — this
  engine's `--keep` already matches a prompt with a picture in it on the rows
  the tower made rather than on the ids, which is half of the mechanism.

  This is not fresh-image acceleration and must never be reported as one. It is
  worth having because asking three questions about one photograph is the
  ordinary case, and because the identity discipline it forces — orientation,
  resize budget, backend, encoder version — is the same discipline any of the
  approximate vision work above will need.

- **The other widths of the integer dot product.** 0.8.9's path is written for
  AVX-512 VNNI and nothing else. A host with `avx_vnni` and no AVX-512 —
  everything from Alder Lake on — has the same instruction at 256 bits; an ARM
  host has `sdot`/`udot` at 128. Both are the same kernel at another width, and
  both wait for the same evidence 0.8.7 waited for on AVX-512: a host with the
  instruction and the headroom to show it.

  `RESEARCH.md` idea 3, the lookup-table execution, belongs here rather than
  ahead of it. It answers the question the integer path already answered, so on
  a host with an integer dot product there is nothing left for it to win; on a
  host with neither instruction it is the only route to the same place.

  *Not in mainline llama.cpp* for the lookup-table form — that is a separate
  fork — but the integer kernels themselves are everywhere there, including AMX,
  which is a fourth width nothing here has looked at.

## Not speed

- **Let the caller choose residency per tensor.** Two thirds of the 2334.8 MiB
  the export maps is never touched by a token — the per-layer embedding table at
  1120 MiB and the two towers at 324 — against the 759.4 MiB a step reads. A
  footprint question, as 0.8.1 established, worth having on a small host, and
  not a decode win however it is scoped.
- **Hold the seam open as the prompt grows.** All nine cases are judged on the
  shipped export, each in its own process, and the longest is 664 ids with two
  pictures and a clip. What has not been tried is a prompt long enough that the
  reference's forward stops fitting even alone — the cost is the reference's,
  and the answer is probably to compare against a cached forward rather than a
  live one.
- **Run two conversations at once rather than one after the other.** The
  sessions are already independent; every kernel under them reaches the model's
  one `pool_group`, which is a fork and join with no queue and one caller's to
  be inside at a time — and 0.8.10 added the attention's two jobs and the gelu's
  to what reaches it, so there is more of the step behind that single gate than
  when this was written. It wants either a pool a session owns — a thread a
  session, and the host's cores to whoever asks first — or a queue in front of
  the one pool, which is the shape a server wants anyway. Neither is worth
  guessing at without a caller that needs it.
- **Add a device handle beside `plane` and a second `back_open`,** so an
  accelerator backend drops in without touching the loader. It is also where a
  backend that repacks a plane at open would keep that packing, which every
  kernel experiment above would want. Written as an accelerator item; it is a
  portability one first.
- **Build under MSVC.** The suite builds clean and passes on Windows with MinGW
  gcc on the scalar, SSE2 and AVX2 backends; `cl` and its `/arch:AVX2` and
  `/arch:AVX512` paths have still only been read.
- **The jpeg frames still refused.** Lossless, differential, hierarchical and
  arithmetic-coded, each another entropy coder or another frame shape rather
  than another branch. None of them is what a caller with a photograph has, so
  this is completeness rather than use.

## Closed, and where to read why

| item | closed by |
| --- | --- |
| Baseline jpeg; wider png; multi-turn chat with several conversations; the prompt cache written and reread | 0.8.5 |
| A cached key row read once for the heads that share it | 0.8.5 |
| The per-group gain mirror (refused: this export's scales are already `F32`, one a row) | 0.8.4 |
| Vectorizing the conformer's score loop (refused: the window is thirteen keys wide) | 0.8.4 |
| Progressive jpeg, `APP14` and twelve-bit samples; a session file that says what it holds; the odd widths read as one word | 0.8.6 |
| The picture attention softmax as a series rather than a call | 0.8.6 |
| The spread's vector path at the odd widths, and the fused dot with it; AVX-512 as a tier above AVX2 | 0.8.7 |
| What the four and eight bit paths wait on at four threads — and the 277 forks and joins that were a third of a token | 0.8.8 |
| A fresh profile of a picture, and four lanes of a batch sharing the row's load | 0.8.8 |
| The eight lane block (refused: 16% of one loop was not worth the engine's last bit; 0.8.9 spent it on something larger) | 0.8.8, settled in 0.8.9 |
| How `app_diff.py` should treat a picture the two sides decode differently (answered: the decoder is not in the comparison) | 0.8.5 |
| Spending fewer instructions in the decode paths — all three widths, both halves of the engine | 0.8.9 |
| What a token spends outside the kernels, a part at a time — and the answer that most of it never was outside them | 0.8.10 |
| The attention scoring and blend across the pool, and the gelu with them | 0.8.10 |
| The logit cap across the pool (refused: measured, and a single fork after the head does not settle) | 0.8.10 |
| The logit cap itself — the series rather than 262144 calls to `tanhf`, on the calling thread still | 0.8.11 |
| The row epilogue's two out-of-line calls, and the `vzeroupper` they dragged through the row loop | 0.8.11 |
| The gelu's own `tanhf`, 215040 calls a step (and the series is the more accurate of the two) | 0.8.11 |
| The bf16 dot product, which had never had a vector path and carries 26.25 MiB of every step | 0.8.11 |
| Why a kernel looks slower on a short row (answered: it does not — `GiB/s` is not comparable across bit widths) | 0.8.11 |
| Eight rows a block instead of four (refused: a wash on every plane, for eight live accumulators) | 0.8.11 |
| Software prefetch ahead of the row loop (refused: 12 to 20% worse on every code plane) | 0.8.11 |
| How a block of rows closes — sixteen folded into one vector instead of four reductions and four rows of scalar | 0.8.12 |
| The batched path's four lanes closed with the same fold, which is where prefill's share of it came from | 0.8.12 |
| Whether the page walk explains the output head's microbenchmark gap (answered: no — the same bytes cost the same with a gigabyte swept in between) | 0.8.12 |
| Whether widening a block of rows can help on its own (answered: no, at any width — the ratio of close to work is fixed by the columns) | 0.8.12 |
| The fork and the join off the mutex — a spinning worker publishes and collects on atomics, and the lock is only what a sleeper is woken through | 0.8.13 |
| A stress test that grinds the pool on both paths and on the handoff between them, with both wakes checked by removing them | 0.8.13 |
| The packed field taken with one bit matrix multiply rather than a variable shift and a mask, at two bits and at four | 0.8.14 |
| Why the output head is slower than anything else per byte (answered: it runs the float kernel — the export's `lm_head.input_activation_scale` is `0.0`, so it can never be on the grid) | 0.8.14 |
| Whether the four bit planes are memory-bound or issue-bound (answered: issue-bound — a quarter off the inner loop is ten to eighteen percent off the phase) | 0.8.14 |
