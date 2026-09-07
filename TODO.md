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

0.8.9 finished the kernels. All three widths the export packs now stream at the
memory's own rate — 25.90, 29.47 and 31.04 GiB/s of codes at four threads
against that 32.18 — so no kernel that reads a code plane has anything left to
give. Decode runs at 12.57 tokens a second of the 41, and prefill at 33.96 on a
288 id prompt.

That leaves two questions, and they are the first two entries below. **Where do
the other 56 ms of a token go?** — nobody has measured it, and it is now two
thirds of the step. And **how does anything get past 41?** — only by producing
more than one token per sweep, which is one idea and it is the second entry.

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

- **Measure what a token spends outside the kernels, a part at a time.** This
  gates everything else on this list and is the largest single unknown in the
  engine.

  Decode at four threads is 12.57 tokens a second, reading 9.63 GiB/s of the
  784.4 MiB a step sweeps. The memory would hand those bytes over in 24 ms and
  the step takes 80. So 56 ms a token — **two thirds of it** — is somewhere the
  kernels are not: the attention, the norms, the sampler, the 277 forks and
  joins, the bands that do not divide evenly, the per-layer embedding lookup,
  the residual adds. 0.8.8 found a third of a token hiding in the fork and join
  and nobody had thought to look; there is twice as much hiding now and the same
  reasoning applies.

  Until this exists, nothing below can be sized honestly, and no new kernel can
  be justified — there may not be one worth writing.

  Cheapest useful form: a compiled-in phase timer around the named parts of
  `session_step`, reported by `bench --verbose`, at one thread and at four. It
  is not a profiler and does not need to be.

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
  be inside at a time. It wants either a pool a session owns — a thread a
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
