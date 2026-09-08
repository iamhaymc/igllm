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

**That ceiling is a host's and not the engine's, and it is the one number here
most often misread.** The third host — four cores of a Xeon at 2.1 GHz,
virtualized, AVX-512 with VNNI and GFNI — sweeps at **49.80 GiB/s** at four
threads, 24.60 at two and 13.04 at one, so its own sweep ceiling is 14.9 ms a
token, or **67 tokens a second**. The engine reaches 28.61 there. Whichever host
a measurement is taken on, quote its sweep beside the step or the ratio means
nothing; 0.8.15 measured this one because no entry above had.

0.8.13 to 0.8.16 took the step floor on that host from 43.44 ms to **34.95** and
decode from 23.02 tokens a second to **28.61**, with `logits` and a greedy
`chat` byte for byte unchanged. 0.9.0 then built the half of speculative
decoding that can be measured without a proposer and found that **the most that
idea can be worth here is 2.2x**, held down by the same output head — so the
sweep ceiling is not the only ceiling, and the entries below are ordered by
which of them unlocks the others. Where those went, largest first: the output head
11.87 ms to 5.99, the four bit planes ten to eighteen percent on the unpack, and
every plane a little on the fork and the join. `CHANGES.md` has each of them.

**0.9.5 took the head off the float kernel, which was the last large thing in
it.** On the reference host — four cores of a Xeon at 2.8 GHz, AVX-512 with
VNNI, a bare four thread sweep of **28.43 GiB/s** measured again there — a plane
the export left uncalibrated now brings a step of the activation's own, and the
head is **7.864 ms to 5.384**, the step floor 48.38 ms to **45.70** and decode
20.67 tokens a second to **21.88**. The batched head came with it and the
speculative ceiling at a block of eight is **2.09x to 2.35x**. Two entries below
close with it: the head's remaining kernel route, refused on ports rather than
on instructions, and the cluster bound, refused on the head's geometry. The mlp
is now 56% of a step against the head's 11.5%, so what is left in this list is
overwhelmingly a memory question and not a kernel one.

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

The order below is the order to take them in, and it is a dependency order
rather than a ranking. 0.9.0 measured what speculative decoding is worth in this
engine and found the ceiling held down by the output head, so the head entries
now come before the speculative one rather than beside it. Where an entry is
only a size, its size is given against a 34.95 ms step floor on the third host.

**Two entries below are out of order on purpose and are worth naming here.**
The first several are the text stack's, and every one of them is either finished
or waiting on a host this project does not currently have — a PMU, or AVX-512
with more bandwidth. Meanwhile 0.9.10 and 0.9.11 measured a multi-modal turn
properly for the first time and found the largest unclaimed number in the file
sitting in the caching, not in a kernel: **a second question about an encoded
picture costs 10.5 s where the same question again costs 0.53**, because a kept
cache is used only when the whole of it is a prefix. That is *reuse a kept cache
up to the longest shared prefix*, and on a host that runs pictures it is the
first thing to take. *Row-blocking the batched float dot product* beside it is a
refusal, and is placed there because it explains why the AVX2 float kernels are
where they are.

- **The feed-forward planes, which are half of every token and are closer to
  the memory than this entry used to say.** 18.96 ms of a 36.9 ms step —
  **52%** — reading 475.3 MiB at 24.44 GiB/s.

  **0.9.2 corrected the ratio this entry was written on, and it halves the
  prize.** The entry read "two point one times its own memory floor", and that
  is the third host's number: a bare four thread sweep there reaches 49.80
  GiB/s. On the reference host a bare sweep is **31.29 GiB/s** — measured again
  in 0.9.2, and 0.8.9's 32.18 on the same machine — so the plane's floor is
  14.83 ms against the 18.96 it takes, which is **1.28 times its floor and 78%
  of a bare sweep**. A plane with no arithmetic in it at all would save 4.1 ms
  of the step here, 11%, and 0.8.9 measured the four bit code path itself at
  29.47 GiB/s in isolation, so the honest ceiling is nearer 17% of the plane.
  Quote a host's own sweep beside this ratio or it means nothing.

  **The chain was the named hypothesis and it is refused, at both widths.**
  0.9.2 built two accumulators a row into `KERN_LEVEL_ROWS_LOOP` — eight chains
  where there were four, bit-identical because integer addition is associative,
  and `logits` byte for byte unchanged to prove the build did it. The mlp phase
  moved from 18.989 ms to 18.959 over three alternating runs, which is nothing.
  A loop within a fifth of what its memory will hand over cannot be
  latency-bound however its chains count.

  **And the same pairing in `KERN_LEVEL_MANY_LOOP` is 15% worse**, which is the
  more interesting half. A batch reads the plane once and multiplies it by every
  lane, so memory stops binding after a lane or two and that loop really should
  be latency-bound — but eight accumulators, two decoded code vectors, four lane
  pointers and the plan's three constants do not fit, and prefill fell from
  78.28 tokens a second to 66.22. **The batched integer path is at its register
  limit, not its latency limit.** Anything written into it has to spend
  registers it does not have.

  What is left, then, is a plane at 78% of this host's memory with its two
  obvious kernel routes closed, and the next thing to try is probably not a
  kernel at all. This export is dense — `num_experts` is null in its
  `config.json`, so there is no routing to exploit — and 35 layers of
  1536 by 6144 gate, up and down is simply the tensor. Either fewer bytes are
  read for a token, which is a sparsity or a residency question and not a loop
  one, or this plane is done. Take a host with more bandwidth than this one
  before reopening it as a kernel: the third host sweeps at 49.80 GiB/s, where
  the same plane sits at 45% of a sweep rather than 78%, and whatever is left in
  the loop would show there and cannot show here.

  **0.9.5 divided the gap, and it is two things rather than one.** The plane's
  own 472.5 MiB were run three ways on the reference host at four threads: swept
  sequentially, walked in the kernel's own pattern — four rows interleaved, the
  same thirty-two byte loads, added instead of decoded — and then by the kernel
  itself. In one quiet run: **sweep 16.4 ms, walk 18.4, kernel 22.6**. So

  - **12% is the access pattern**, and it is not a bug. Four rows in flight is
    what shares the staged levels and the epilogue between them, and reading
    them costs more than reading one stream. It is the price of the row block,
    not something left on the floor.
  - **the other 19% is issue**, at about 1.4 ms per port-0-or-5 uop in the inner
    loop. Removing the shift and the mask and keeping the dot product costs
    19.1 ms; keeping them and dropping the dot product costs 21.9. Three uops
    per sixty-four codes a row — `vpsrlvd`, `vpandd`, `vpdpbusd` — and each is
    worth about the same.

  **Which says the one instruction worth removing is already removed where the
  host allows it.** `vgf2p8affineqb` does the shift and the mask as one, and
  0.8.14 built that path; it is gated on GFNI, which the reference host does not
  have and the third host does. On a host without GFNI there is nothing cheaper
  in AVX-512BW — the two uops are a variable shift and a mask, and no single
  instruction in that subset does both — so this half of the gap is a property
  of the instruction set and not of the loop.

  The same measurement at **one** thread gives the same ratio (kernel at 70% of
  its sweep, 79% of its walk), so none of it is the fork, the join, or the
  threads landing unevenly.

  What is not worth retrying: the bit width (0.8.11), eight rows a block
  (0.8.11, and again in 0.8.16 on the other path), software prefetch (0.8.11),
  the page walk (0.8.12), the accumulator chain at either width (0.9.2), and the
  access pattern or the unpack on a host without GFNI (0.9.5).

- **The output head, which used to run the float kernel and now runs the
  integer one.** 96 MiB and 12.2% of everything a decode step reads, and until
  0.9.5 the slowest plane in the step per multiply-add for a reason no kernel
  change could reach.

  0.8.14 settled what three versions of this entry could not. `kern_level_ready`
  required `sheet->enter_gain > 0`, the plane's `input_activation_scale`, and
  **the export ships `lm_head.input_activation_scale` as `0.0`** where every
  other projection has a real one — so the head fell to `kern_row_code`'s float
  spread on every token, by construction, and 0.8.11 through 0.8.16's work on
  the integer path could never have moved it.

  **0.9.5 gave the head a step and took the integer path, and that is the entry
  closed.** `kern_level_pick` takes the lane's own largest magnitude over the
  127 levels a signed byte carries, so the step is per token and per lane rather
  than the export's; `kern_level_stage` rounds onto it where before it verified
  a grid the caller was already on. On the reference host the head is **7.864 ms
  to 5.384**, 51.2 G multiply-adds a second to 74.8, the step floor 48.38 ms to
  45.70 and decode 20.67 tok/s to 21.88. The batched head came with it, so a
  block's marginal lane is 21% cheaper and the speculative ceiling at a block of
  eight is **2.09x to 2.35x**.

  The rounding costs 0.04 root mean square on the row and 0.49 at worst, over
  the whole vocabulary, which moved the top token once in 384 greedy steps — at
  a gap of 0.023. So the sweep is taken on the grid and the decision is not:
  `session_head_true` scores the best sixty-four rows again on the float path,
  a four thousandth of the plane, and the top token and the top five then match
  the float build on every one of those 384 steps and the greedy text matches
  byte for byte on all eight prompts. `CHANGES.md` 0.9.5 has every number.

  **What is left, and it is now the smaller half.** The head is 5.38 ms against
  a bare four thread sweep of its own 97 MiB at 3.41 ms, so it is at 63% of this
  host's memory where it was at 43%. Whatever remains is worth under 2 ms of a
  45.7 ms step, and the plane is no longer the outlier the rest of this entry
  was written about.

  **The float loop's broadcast is refused, and refused on ports rather than on
  instructions.** The route was to replace three of the loop's sixteen
  instructions per sixty-four codes with one `vbroadcasti32x4`, at the cost of
  staging the activations in the unpack's order. It was built and measured
  against the shipped loop over the real head: 7.763 ms to 7.880, a wash. The
  three instructions it removes are `vpbroadcastd` from memory, which retire on
  the **load** ports; the loop is bound by ports 0 and 5, where the variable
  shift, the `vpermps` and the multiply-add sit, and twelve of those uops per
  sixty-four codes is what it was before and after. **Counting instructions is
  not counting ports** — which is the second time this entry has had the loop
  under test not be the loop that was binding. Anything aimed at the float head
  now has to take work off ports 0 and 5; 0.9.5 took the whole loop off them
  instead.

  What is ruled out, and should not be tried again: the row epilogue (0.8.11,
  and it is in the wrong path anyway); eight rows a block instead of four
  (0.8.11, a wash); software prefetch of the code stream (0.8.11, 12 to 20%
  *worse* on every code plane — read the numbers before having the idea);
  sixteen rows closed together (0.8.12, wrong path); the page walk (0.8.12 — the
  same 96 MiB costs the same with a gigabyte swept in between); the affine take
  (0.8.14, wrong path); and the broadcast (0.9.5, above). Do not reopen any of
  it on a GiB/s comparison across bit widths, which is not a comparison at all.

- **Stop scoring 262144 rows to pick one — closed by measurement, and it is a
  refusal.** `RESEARCH.md` idea 10 was to cluster the head's rows once at load,
  bound each cluster by `<centroid, a> + radius * ||a||`, and discard a cluster
  whose bound cannot beat the best candidate so far.

  The entry named the one thing to check before building any of it, and 0.9.5
  checked it. **The bound prunes nothing, and the reason is the geometry rather
  than the clustering.**

  A real activation, the exact logits beside it, and the rows bucketed by a sign
  signature over random directions — a clustering cheap enough to be a load-time
  cost, which a k-means over 262144 rows is not:

  | cells | mean radius | radius over the mean row norm | rows the bound keeps |
  | --- | --- | --- | --- |
  | 256 | 0.988 | 1.054 | 100.0% |
  | 4096 | 0.921 | 0.982 | 100.0% |
  | 16384 | 0.784 | 0.837 | 99.6% |

  Sixty-four times more cells moved the radius from 1.05 of a row norm to 0.84,
  and the bound needs it under **0.134** — the top logit over `||a||`, measured
  over 32 real greedy steps, where it ranged 0.097 to 0.201. The plain per-row
  Cauchy-Schwarz bound is the floor and behaves as this entry predicted:
  `||w|| ||a||` is **25.8x** the value it bounds, and it keeps every row.

  **And the refusal holds for every clustering there is, not just that one.** A
  cell holding two rows has radius at least half the distance between them, so
  the best radius any clustering with more than one row a cell can reach is half
  the nearest neighbour distance. Over a sample of 256 rows against all 262144,
  the nearest neighbour sits at **0.837** on average against a mean row norm of
  0.937 — half of which is 0.418, three times looser than the 0.134 the bound
  needs — and only **1.2%** of rows have a neighbour within twice what the bound
  needs. 262144 rows in 1536 dimensions are very nearly mutually orthogonal, and
  no cell of more than one of them can be tight enough.

  So the entry is closed, and closed the way its own rule said to close it: stop
  if the bound needs most of a row to be useful. It needs all of one.

  What it was worth is also gone, and by the better route. This was the enabler
  for the speculative entry's step 2 and for the head entry above, and both of
  those wanted it because the head cost five times more per byte than any other
  plane. 0.9.5 took 46% of that instead, by putting the head on the integer path
  — so a bound would now be pruning bytes that are three times cheaper than when
  this entry was written, on a plane that is 11.5% of a step rather than 16%.

  Do not reopen it on this export. A checkpoint whose head is *tied* to the
  embedding table is a different question — there the rows are already read for
  the embedding lookup — and so is one whose vocabulary is small enough that the
  clusters could be tight; neither is this one.

- **Produce more than one token per sweep of the weights.** The only idea here
  that can pass the sweep ceiling at all, because it is the only one that
  changes the ratio of tokens to sweeps rather than the cost of a sweep.
  `RESEARCH.md` idea 1 in full; the short form is a cheap proposer guessing a
  short block and the real model checking the whole block in one batched pass,
  keeping the guesses it agrees with.

  *llama.cpp has this three ways over* — a draft model
  (`common/speculative.cpp`), lookahead decoding (`examples/lookahead`), and
  n-gram/prompt lookup with no second model at all (`common/ngram-cache`,
  `ngram-map`, `ngram-mod`). So the idea is not in doubt and neither is the
  payoff there; what was unknown was this engine's verification cost, and 0.9.0
  measured it.

  **It is built and it ships — 0.9.3 — and what is left of the entry is the
  ceiling rather than the feature.** `igllm guess` runs a greedy continuation
  with three proposers, one always right, the n-gram scout that ships, and one
  always wrong, and holds all three to the plain run's token stream. On the
  shipped export at four threads, on an answer that quotes its prompt:

  | block | proposer | committed a round | vs plain |
  | --- | --- | --- | --- |
  | 4 | oracle | 4.00 | 2.02x |
  | 4 | n-gram | 2.67 | 1.69x |
  | 8 | oracle | 8.00 | 2.11x |
  | 8 | n-gram | 3.69 | **1.80x** |
  | 8 | null | 1.00 | 0.30x |

  **The ceiling is still about 2.2x** — that is a proposer that is never wrong —
  and the scout reaches **85% of it** where prompt lookup applies. On free
  generation it draws nothing and costs 0.92 to 0.95, which is why `--guess` is
  a flag.

  The reason the ceiling is 2.2 and not 6 is what is left of this entry. A block
  shares the weight *sweep* across its lanes and cannot share the *arithmetic*,
  and this engine is arithmetic-bound: an extra lane cost about **13 ms** against
  a plain step's 38 when 0.9.0 measured it, and 0.9.1 took it to **10.6** by
  giving the batched head the one lane head's unpack and chain depth. **The head
  was six of the original thirteen and is now about three.** Halve what is left
  of the marginal lane and the ceiling goes toward 4x — which would take the
  scout's 1.80x with it, and would move free generation from a cost to a wash.
  The head entry above is therefore still this entry's ceiling, and its second
  route — not scoring 262144 rows to answer one question about one row — is the
  one that has not been tried.

  **The order to finish it in.**

  1. **Done, 0.9.0: the verify side.** All-position logits out of `session_pass`
     (`PASS_LOGIT_ALL`), a cache that can be put back (`session_guess`,
     `session_guess_keep`, `session_guess_limit`), and the bracket above. The
     undo is held byte for byte by `test_guess` on the synthetic checkpoint,
     whose four row window makes every block lap the ring; the bug the old entry
     warned about — trusting `fill_count` — was written deliberately and the
     test fails on it.

  2. **Done, 0.9.1 and 0.9.5: make the batched head cheap.** Three routes were
     named and all three are now settled. `kern_dot_code_many` gave the many
     lane path 0.8.15's `vpermps` lookup and 0.8.16's chain depth at once, and
     the marginal lane at a block of eight went 12.44 ms to 10.56 — a round 8 to
     13% cheaper. **0.9.5 then took the head off the float kernel entirely**: a
     step of each lane's own puts the whole block on `vpdpbusd`, the marginal
     lane at a block of sixteen is 22.76 ms to 18.08 on the reference host, and
     the oracle ceiling is **2.09x to 2.35x at a block of eight, 2.18x to 2.40x
     at sixteen**. The third — verification does not need the whole
     distribution, only whether the proposed id is the argmax — wanted the
     cluster bound in the entry above, and that entry is now closed as a
     refusal: the bound cannot be made tight enough on a head this nearly
     orthogonal. So this step is finished, and what remains of the ceiling is
     not in the head.

     **And 0.9.5 divided a round, which says where the lane actually goes.**
     `igllm guess --verbose` prices the marginal lane part by part, by
     subtracting the oracle's narrowest block from its widest. On the reference
     host, a block of 2 against a block of 16 on a 49 id prompt:

     | part | ms at 2 | ms at 16 | ms a lane | share of the lane |
     | --- | --- | --- | --- | --- |
     | mlp | 36.782 | 160.924 | **8.867** | **49.3%** |
     | final norm, head | 7.658 | 37.061 | 2.100 | 11.7% |
     | score, softmax, blend | 4.076 | 31.862 | 1.985 | 11.0% |
     | q k v | 5.597 | 21.824 | 1.159 | 6.4% |
     | attn out | 5.079 | 21.037 | 1.140 | 6.3% |
     | ple feed | 3.224 | 15.843 | 0.901 | 5.0% |
     | named, in all | 67.377 | 319.192 | **17.987** | 100% |

     **So the head is no longer the ceiling and the mlp is.** The entry above
     used to read that the head was six of the original thirteen milliseconds
     and 0.9.1 made it three; it is now **2.1 of 18.0**, and half the lane is
     the feed-forward. Nothing above this line needs re-deriving — take the
     table.

     **And the mlp's marginal lane is nowhere near its own instruction floor,
     which is the open question this entry now turns on.** A lane of the mlp is
     991 M multiply-adds; `vpdpbusd` retires sixty-four of them, so that is 15.5
     M instructions, and `KERN_LEVEL_MANY_LOOP` issues five port-0 uops per four
     lanes per block — about 2 ms over four threads. The measured lane is
     **8.9**, four times that.

     0.9.5 measured the two obvious explanations and neither is it:

     - **The epilogue is not it, and 0.9.9 has now taken it anyway.** The close
       accumulated straight into the destination, a group at a time and strided
       a whole lane apart — sixteen scattered read-modify-writes per group of
       every row. The totals now live in a local and reach the destination once
       a row, which is `kern_blend_rows`' move and moves no number: prefill
       **101.10 to 105.53 tokens a second**, the marginal lane **10.31 ms to
       9.19**, the oracle at a block of sixteen 2.54–2.63x to **2.67–2.82x** and
       the scout 1.83x to **2.02x**. That is 1.1 ms of 10.3, which is what this
       note predicted, and it leaves the gap below where it was.
     - **The staging stride was not it either.** Every lane's levels used to be
       laid down `desk->level_limit` apart, which is the widest code plane the
       model binds — `ple embed`'s 8960 columns — so sixteen lanes of a 1536
       column plane sat 8960 bytes apart where the 24 KiB they actually read
       would have fitted in the first level cache side by side. 0.9.5 stages at
       the plane's own width instead. In isolation that is **6.6%** of the
       batched feed-forward; end to end on this host it is a wash inside the
       noise, and it ships because it is the same work with better locality and
       costs nothing, not because it was measured to pay.

     So four fifths of the batched lane is still unexplained, and it is now the
     largest single number in this list — 8.9 ms a lane against a 45.7 ms step,
     with the ceiling in the table above riding on it. **Take a hardware counter
     to it before writing another loop**: the four hypotheses that can be
     reasoned about from the source have now all been measured, and all four
     came back small — the epilogue was the last of them, and 0.9.9 took it for
     a tenth of the lane rather than the four times the gap needs.

     **And that instruction is now the blocker rather than the advice.** The
     host these figures were taken on exposes no performance counters at all:
     `/sys/bus/event_source/devices` holds `breakpoint`, `msr`, `power`,
     `software`, `tracepoint` and `uprobe`, and no `cpu`, so `perf stat` has
     nothing to count and neither would anything else. This entry needs a host
     with a PMU before it can move, and that is the whole of what it is waiting
     for.

  3. **Done, 0.9.3: the n-gram proposer.** `app_scout` asks what followed the
     last time this stream said what it has just said — a growing array of ids
     and a backward scan, no second model and no training. It runs as a third
     row of `igllm guess` beside the oracle and the null, so it is held to
     committed tokens per millisecond against the bracket rather than to an
     acceptance rate. On an answer that quotes its prompt it reaches **1.80x at
     a block of eight against a ceiling of 2.11x** — 85% of everything a
     proposer that is never wrong could give, at 95% of guesses accepted. On
     free generation it draws nothing at all and costs 0.92 to 0.95 of the plain
     rate, so **it ships behind a flag**, which is what this step said to do if
     it did not clear break-even there.

     Two things settled in passing and worth not re-deriving. The shortest reach
     it will match on is **two, not one**: a single id matches somewhere in
     almost any stream, so a reach of one draws nearly every round and is right
     almost never — and dropping it is better on the quoting workload *and* on
     free generation at once. And where the scout proposes nothing the round is
     an ordinary `session_step` rather than a block of one lane, which is the
     other half of why the flag is nearly free when it does not help.

  4. **Done, 0.9.8: sampling, and the choice this step posed was a false one.**
     The step read that an n-gram proposer has no `q` to divide by, so the
     feature was either greedy-only or wanted a proposer that carries a
     distribution. It has one: the scout names a token, so `q` is a **point
     mass** on it. Put `q(t) = 1` into the modified rejection rule and both
     halves collapse — accept with probability `p(t)`, and on a rejection draw
     from `p` with the guess taken out and the rest renormalized. Nothing is
     approximated and no second model is carried.

     `p` is the caller's whole taste rather than a bare softmax, and each lane
     is shaped against the history it would have if the guesses before it were
     kept. `--guess` above `--heat 0` used to be refused outright, and the
     default taste is `--heat 1`, so the flag was unusable unless asked for.

     **Acceptance under a temperature is the model's own certainty**, where
     greedy acceptance only asks whether the guess was the argmax — so it pays
     better where the model is sure and worse where it is not. Repeating a
     passage back verbatim at `--heat 1` is 33.55 tok/s to **68.32 at a block of
     four and 79.15 at eight**, 100% of 63 guesses kept and 96% of 77, which is
     *above* the greedy path's 1.69x and 1.80x. Free generation costs about 6%,
     the same place greedy sits. So it stays a flag.

     A greedy block is still byte for byte a greedy run. A sampled one is not
     and cannot be: a round takes one draw where its guess is accepted and two
     where it is not, so the streams diverge from the first rejection. The
     distribution is equal, which is what speculative sampling guarantees, and
     `CHANGES.md` 0.9.8 says how that is tested rather than asserted.

     **What is left of it is a proposer that carries a real `q`.** The point
     mass is the strongest possible proposal and therefore the harshest: it
     stakes everything on one token, so acceptance can never exceed `p(t)`. A
     proposer offering a distribution — a small draft model, or the scout's own
     counts turned into one — is accepted with `min(1, p/q)`, which can be 1
     over a whole region rather than only at a near-certain token. That is the
     route to raising acceptance under a temperature, and the rule it needs is
     already built and already tested; only the proposer is missing.

  5. **Done, 0.9.3: the plumbing.** `--guess <lanes>` on `chat` and `complete`,
     the block path inside `main_serve`, and a tally line counting committed
     tokens against rounds and accepted guesses against drawn. `session_guess`
     now counts into the decode tally as well — the weights once and the cache
     once a lane — so `decode tok/s` and `reads MiB a token` describe a guessed
     run rather than reading zero.

     **And 0.9.4 carried it into `chat --loop`,** which is where prompt lookup
     belongs: the scout is the conversation's rather than the turn's, so a
     follow-up question about the same document has both the document and the
     answer before it. A passage planted in one turn and asked for back in the
     next is **5.56 s to 4.33** end to end, and the second turn commits 3.08
     tokens a round at 82% of guesses kept where a per-turn scout would have had
     nothing at all.

     What is left is small and named: a prompt whose last id is a soft token
     from a tower takes the plain loop whatever the flag says, because a block
     cannot carry an embedding row.

  **One thing to watch that 0.9.0 found and did not settle.** A batched pass
  answers every lane the same way — one lane off the grid puts the whole batch
  on the float path — while a lane stepped alone is judged alone. (Since 0.9.5
  the head cannot be that lane: a plane with no step of its own gives every lane
  one of its own, so the head's block is always on the integer path. Every other
  plane is calibrated and can still refuse.) So
  a block and a sequence of steps can take different kernels and sum in a
  different order, and an accepted token is the token the model would have
  produced rather than the same arithmetic that would have produced it. Over 32
  tokens on two prompts the streams matched exactly; that is evidence and not a
  proof, and a long horizon is where it would first show. If `igllm guess` ever
  prints `DIVERGES`, this is what it means and it is not necessarily a bug.

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

- **What is left of the tower's own attention.** 0.9.6 retook the profile this
  entry asked for and took the largest thing in it. Scoring and the blend were
  **52.7% of a picture**; a block of four queries sharing every byte both loops
  read made the phase **1.9x** and a picture 10.64 s to 8.10 s, at the same
  arithmetic in the same order — `logits` after a picture is byte for byte what
  it was. `CHANGES.md` 0.9.6 has the division and every number.

  **The tiling itself is refused, and it is worth knowing why before anyone
  reopens it.** `RESEARCH.md` idea 11's prize is the score matrix it stops
  materializing, which at 2304 patches is 2304 by 2304 — but a band here holds
  **one score row, not the grid**, so that memory was never spent and the idea
  arrives with nothing to save and a summation-order change to pay. llama.cpp's
  `ggml_compute_forward_flash_attn_ext_tiled` is written against a runtime that
  does materialize it. Do not port it on the strength of that.

  **What is actually left is the arithmetic, and the loose axis is spent.**
  After 0.9.6 the phase is 34 to 41% of a picture and the feed-forward is the
  larger part of a tower — *on the host 0.9.6 measured.* **0.9.11 found a host
  where the dense pair is about a sixth of the encoder**, by answering a turn
  with the tower skipped and fitting `a·n + b·n²` over seven grid sizes: 3.25 ms
  a patch plus a term worth 17.8% at 2340 patches, on four AVX2 cores with no
  AVX-512. Some of the gap to 34–41% is the denominator — that figure is against
  a picture and this one against the encoder, and half a picture is not the
  encoder at all — but not all of it, and the two hosts should not be assumed to
  agree. **Take the split on the host before taking the work**, with the picture
  file as the instrument, and prefer the patch budget where the phase is small. Scoring ran 6.7 G multiply-adds a second and the blend
  8.0, against a 256-bit FMA peak of 44.8 a core; both are now roughly twice
  that and still a long way short. Three things are known about what is left:

  - **The fused multiply-add in the blend is worth another 0.6x and was
    refused, not missed.** `kern_blend_rows_many` keeps a multiply and an add
    because dropping the intermediate rounding would stop a picture's rows being
    the rows that ship. Measured: 2.2x fused against 1.4x not, on the AVX-512
    host. It is a decision about output rather than a kernel question, and the
    test that holds the kernel bit for bit fails on it deliberately. Taking it
    means saying so in `CHANGES.md` and re-taking the media parity run.
  - **It is also the worse kernel on a plain AVX2 host** — 1.28x against 1.32x —
    because sixteen accumulators and four value registers do not fit in sixteen
    registers. Anything wider than `KERN_GRID_LANE` has the same problem, so a
    wider block needs the register file asked for rather than assumed.
  - **The key axis is untouched and is the one left.** Four queries share the
    key row; nothing shares the *query*. A block of keys against a block of
    queries is the register-blocked matrix product this loop really is, and it
    is the only route here that does not need the summation order.

  The softmax is 6.6% of the phase and not worth opening.

- **Fewer patches, before the pooling. The cheap half is built — 0.9.10 — and
  what is left is the merging.**

  The tower is 16 layers of width 768, 12 heads, 16-pixel patches, and the 3×3
  pooling that turns the patches into soft tokens happens **after** the encoder.
  So every patch is paid in full and the pooling saves nothing but the text
  stack's share. That is what makes both halves of this entry worth having.

  **The budget ships.** `model_image_budget` and `--image-tokens <n>` cap what a
  picture may cost, by moving the resize and nothing else — the pool geometry,
  the positions and the emitted row count are all still exactly what they were
  for a grid of that size, and the cap is inside the picture store's identity so
  that two budgets over one photograph cannot collide on one entry. On a fourth
  host (i5-7600K, AVX2, no AVX-512) a 768×512 notice is a **19.39 s turn at the
  checkpoint's 280 rows and 3.25 s at a budget of 40**, transcribed exactly at
  both — 5.97x end to end, 7.81x on the tower alone. `CHANGES.md` 0.9.10 has the
  latency curve and both quality curves.

  Three things it found that the rest of this entry should be read against.

  - **The tower is nearly linear in patches**, not quadratic: 7.4 to 8.0 ms a
    patch flat across 108 to 2340 patches, which puts the dense attention pair
    at about **8% of a picture at the full grid on that host**. The entry above
    records scoring and the blend at 34 to 41% of a picture on the reference
    host, and the two do not agree. Whichever is right, on a host like this one
    the per-patch work is nearly the whole of a picture and the attention is
    not — so **fewer patches is worth more here than any attention kernel**, and
    a kernel aimed at the attention should be measured on the host it is meant
    for before it is written.
  - **The two halves of the quality curve part company by about a factor of
    two.** Reading a six line notice survives to 35 rows and breaks at 24;
    recognising a scene of four objects is still right at 12. So a caller that
    knows which question it is asking can spend very differently, which is the
    argument for the knob being a knob rather than a default.
  - **Below about six rows the model reports no picture at all** rather than a
    coarse one — *"Please provide the picture you are referring to."* Any policy
    that picks a budget automatically has to hold off that floor, and cannot
    detect having crossed it from the answer.

  **What is left is `RESEARCH.md` idea 5**: merge redundant patches inside the
  encoder after the first few blocks, keeping enough provenance to reconstruct
  the pooling contract. That is unexplored ground on CPU — there is no merging
  or pruning pass in llama.cpp's `clip.cpp`, and the one mention of token
  merging there describes a model variant whose own convolution does it. It is
  strictly harder than the budget and it is worth less than the budget was,
  because a caller that will accept fewer patches can now simply ask for fewer.
  Its case is the caller that will not: full resolution where the picture needs
  it and merging where it does not, inside one pass.

  *llama.cpp has the budget knob* — `image_min_tokens` and `image_max_tokens` in
  `tools/mtmd/mtmd.h`, read from metadata and overridable by the caller, with no
  policy that escalates after a cheap pass. 0.9.10 does not have one either, and
  deliberately: a fast path that falls back to full resolution half the time is
  not a fast path, and nothing here knows in advance which half a picture is in.

  **And the curves above are a shape rather than a study.** Two synthetic
  rasters written for the purpose, one prompt each, greedy. A photograph of a
  page is the case that decides what a caller should ask for, and it has not
  been measured — that, and an escalation policy that could be trusted, are what
  a second pass at this entry would be.

- **Give a picture's rows a life beyond one process.** 0.9.6's entry above is
  built and ships — 0.9.7 — and this is what it left.

  The store is four pictures on the model, keyed on the decoded samples and the
  tower configuration by two independent mixes, and it takes the case the old
  entry named: a photograph, a question, another conversation, the same
  photograph, another question is **53.53 s to 39.72 s**, byte for byte the same
  two conversations. A miss costs under 2%. `CHANGES.md` 0.9.7 has the identity
  and what is deliberately outside it.

  **The word "process" is closed — 0.9.11.** `--image-keep <path>` reads the
  store at the start of a run and writes it at the end, and the three things
  this entry required before a file could be safe are all in it: the backend and
  `desk->level_live` beside it in the mark, a `MEDIA_KEEP_VERSION` bumped by
  hand, and the path and the eviction left to the caller exactly as `--keep`
  leaves them. A refused file leaves the store alone and the run says so; a
  truncated one is refused whole rather than read half. A repeat run of a
  picture is **19.60 s to 10.36 s, 1.89x**, and with a budget beside it two
  turns about one photograph are **39.44 s to 5.58 s, 7.07x**. `CHANGES.md`
  0.9.11 has the rest.

  **And it turned out to be the instrument this list needed more than the
  feature.** Answering a turn with the tower skipped is what separates the
  encoder from the text stack's prefill of the soft tokens, and nothing before
  it could. Doing that across seven grid sizes gives the tower's cost law
  directly, and it corrected a number in the entry above by a factor of two.
  **Half of what a picture costs is not the tower at all** — 9.25 s of 19.60 on
  that host is the encoder and 9.46 is the 260 soft tokens going through the
  text stack, at a flat 36.5 ms each. Nothing in this file had that split.

  What is left of the entry is small and is not the file. A picture's rows are
  reusable across processes now; a *clip's* are not, and the audio tower has no
  store at all, in memory or on disk. Whether that is worth having is a question
  about how often the same clip is asked about twice, and nobody has asked it.

  *llama.cpp has a narrower form of the in-memory half.* Its server pushes a
  placeholder for an encoded media chunk into the slot's prompt tokens — "the
  chunk is already in the KV cache at this point, so we don't need to keep its
  data around" (`tools/server/server-context.cpp`) — so a picture is reused by
  an ordinary **prompt-prefix match within one slot**, keyed by the caller's
  bitmap id. That covers the follow-up turn and nothing else: not the same
  picture in another conversation, not with different text in front of it, not
  after a restart. There is still no content-addressed store of post-projector
  rows anywhere in that tree, and no persisted one either.

  None of this is fresh-image acceleration and none of it must ever be reported
  as one.

- **Row-blocking the batched float dot product — measured, and refused on the
  register file rather than on the loop.**

  Every single-lane path in this engine blocks rows: `KERN_ROW_WIDE`,
  `KERN_ROW_BLOCK`, `KERN_CODE_BLOCK`. The batched one does not —
  `kern_mat_vec_band`'s last branch calls `kern_row_code_many` a row at a time —
  and that asymmetry looks like something left on the floor. It is not, on a
  host with sixteen vector registers.

  Why it looked worth taking. `kern_dot_real_many` holds one row against four
  lanes with two accumulators a lane, stepping sixteen: per sixteen elements
  that is two row loads, eight lane loads and eight multiply-adds. Ten loads
  against eight FMAs, on a host with two of each port, so the loads are the
  ceiling and the arithmetic waits. Blocking a second row shares the lane loads
  between the rows and should put the loop on its arithmetic instead.

  Measured on the fourth host, one thread, a 256 column group over 4096 rows,
  minimum of fifty runs, with every lane's total read afterwards so that nothing
  can be eliminated — **the first version of this measurement was 1.8x and was
  wrong for exactly that reason**, two of the four lanes being dead code:

  | shape | G multiply-adds a second | against the shipped | order |
  | --- | --- | --- | --- |
  | one row, four lanes, two accumulators, step 16 | 27.0–27.4 | — | shipped |
  | two rows, four lanes, one accumulator, step 8 | 36.5–36.7 | **1.33x** | **different** |
  | two rows, two lanes, two accumulators, step 16 | 28.0–28.5 | 1.04x | shipped |

  **The only row block that pays changes the summation order, and the one that
  preserves it does not pay.** Preserving the chain costs two accumulators a
  row-lane pair, so two rows by four lanes wants sixteen accumulators and the
  register file has sixteen registers in all — before the row and lane vectors.
  Two rows by two lanes fits, and halves the lanes, which puts the row loads
  straight back: four row loads and four lane loads for half the work.

  So this is the same wall the tower's blend entry hit — *"sixteen accumulators
  and four value registers do not fit in sixteen registers"* — and it is worth
  stating once in general: **on AVX2 the register file, not the loop, is what
  limits every batched float kernel in this engine.** Anything wider needs the
  registers asked for rather than assumed.

  Two things left, and neither is a loop.

  - **1.33x is not nothing, and it is a decision about output rather than a
    kernel question** — the same shape as the fused multiply-add refused in the
    tower's blend (0.9.6). Taking it means prefill logits stop being byte for
    byte what they were. It is not taken here, and it should not be taken
    quietly if it ever is.
  - **On AVX-512 the order-preserving form fits.** Thirty-two registers hold
    sixteen accumulators, four row vectors and two lane vectors with room over,
    so two rows by four lanes at the shipped chain is available there and is not
    here. **This is a hypothesis and not a measurement** — the host it was
    reasoned on has no AVX-512 — and it is worth an hour on a host that does,
    with one caveat: where VNNI carries most planes onto the integer path, the
    float many path is a smaller share of a step than it is on a host without.

  This microbenchmark is contiguous rows in cache, not the engine, and it says
  nothing about how much of a phase the inner loop is. What it does establish is
  the ranking, and the ranking is a refusal.

- **Reuse a kept cache up to the longest shared prefix, not only where it is
  the whole prompt.** The largest measured win left in this list on a
  multi-modal host, and it is worth more than anything above it that is not
  already built.

  0.9.11 made the split visible and this is what it exposed. A picture's soft
  tokens sit at the **front** of the prompt and the question sits behind them,
  so a second question about the same photograph shares every expensive id and
  differs only in the cheap ones. On the fourth host, a 768×512 notice at 260
  soft tokens:

  | run | seconds |
  | --- | --- |
  | the same question again, `--image-keep --keep` | **0.53** |
  | a different question, `--image-keep --keep` | 10.53 |
  | a different question, `--image-keep` alone | 10.52 |

  The second row is the entry. **The session file buys nothing at all when the
  question changes**, and what it fails to save is the 9.46 s of prefilling 260
  soft tokens that the new question shares with the old one — 36.5 ms an id, and
  not one of them has moved.

  `main_keep_prime` already computes the shared prefix, and its own comment
  describes the behaviour this entry wants: *"What is reused is a prefix and not
  a match."* Then the next line throws it away — `if (same_count != held_count
  || held_stamp != stamp_value) same_count = 0;` — so a held cache is reused
  only when the whole of it is a prefix of the new prompt, which is the case
  where the question did not change. The prefix is computed and then refused.

  **Four things stand between here and it, and the last two are the reason it
  was not simply done.**

  - **The stamp cannot be checked at a prefix.** `main_keep_stamp(reel, k)`
    folds the media rows over the first `k` ids, and the file stores it only at
    the length that was saved, so there is nothing to compare a shorter fold
    against. Ids alone cannot stand in for it: two pictures lay down the same
    placeholder ids, which is why the stamp exists. The fix is to store the fold
    over the media rows beside the id index just past the last of them, and to
    accept a prefix only at or past that index — where the media contribution is
    complete and fixed, one stamp answers for every prefix length.
  - **The echo history has to be cut with it.** `echo_count` and `echo_room`
    are the repetition penalty's memory and are as long as the cache.
  - **The ring may have turned over.** A layer whose `cache_span` is shorter
    than the held prompt holds its rows at slots whose meaning depends on
    `fill_count`, so moving `fill_count` back does not move the rows back with
    it. Either refuse a prefix wherever anything has turned over — which the
    common case is nowhere near — or teach the file to record the ring's base.
  - **The cache peaks cannot be un-maxed.** `key_peak` and `value_peak` are
    running maxima over everything ever written, and truncating the cache leaves
    them describing rows that are no longer in it. Under `--cache 8` they scale
    the quantization, so a truncated session would quantize differently from a
    fresh one and the run would stop being byte for byte — which is the property
    this engine holds every cache path to. Recompute them over the kept rows, or
    refuse a prefix under a quantized cache and say so.

  None of these is hard; together they are a careful change to the one part of
  the engine where a mistake is silent, and it should be taken with the
  reference comparison available rather than without it. `session_guess_keep`
  is the nearest precedent for putting a cache back, and the test that holds it
  — a window narrow enough that every block laps the ring — is the shape the
  test for this wants.

  *llama.cpp does exactly this and calls it the same thing*: its server matches
  a new prompt against a slot's cached tokens and keeps the common prefix, which
  is how a follow-up question about an encoded picture is cheap there. This
  engine has the harder half of it already — the stamp that knows a picture is
  the same picture, which a token comparison cannot.

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
| The two bit float loop's mask and widening as one `vpermps` against a repeating table, in the loop the output head actually runs | 0.8.15 |
| Why the output head was slower than its own port count (answered: two accumulators over ninety-six vectors — the row waited on its own chain) and the four row block that fixed it | 0.8.16 |
| Eight rows a block in the float path (refused: worse on the one plane it is for, as it was on the integer path) | 0.8.16 |
| All-position logits out of `session_pass`, as one batched head rather than a loop of single ones | 0.9.0 |
| A cache a rejected block can be taken out of — the bound that makes it a fixed scratch, and the byte-for-byte test that holds it | 0.9.0 |
| What speculative decoding is worth in this engine, bracketed by a proposer that is always right and one that is always wrong (answered: 2.2x at best, break-even near 40% acceptance, and the output head is half the marginal lane) | 0.9.0 |
| The batched output head's unpack and its chain depth — the two things 0.8.15 and 0.8.16 gave the one lane path and could not reach the many lane one | 0.9.1 |
| The n-gram proposer, measured against the bracket rather than by acceptance rate, on the two workloads that differ | 0.9.3 |
| The `--guess` flag, the block path inside `main_serve`, and a block counted into the decode tally | 0.9.3 |
| A scout that belongs to the conversation rather than to the turn, so `chat --loop` has prompt lookup across turns | 0.9.4 |
| Whether a proposer should match on a single id (answered: no — better on both workloads at once without it) | 0.9.3 |
| The mlp's accumulator chain, which the entry named as the first thing to check (refused: a wash on the one lane block, 15% worse on the batched one, and the plane is at 78% of this host's bare sweep) | 0.9.2 |
| What a bare sweep of the reference host actually is, against the third host's number the mlp entry had been reasoning from | 0.9.2 |
| The output head on the integer path — a step of the activation's own where the export gave the plane none, and the best sixty-four rows scored again on the float path so the decision is not the rounding's | 0.9.5 |
| The float head's remaining instruction, one `vbroadcasti32x4` for three `vpbroadcastd` (refused: a wash — the three it removes are load-port uops and the loop is bound by ports 0 and 5) | 0.9.5 |
| Whether a cluster bound can prune the head's 262144 rows (answered: no, and for every clustering rather than the one tried — the mean nearest neighbour is 0.89 of a row norm and the bound needs 0.14) | 0.9.5 |
| A speculative round divided into named parts, and the marginal lane priced part by part by subtracting the narrowest block from the widest | 0.9.5 |
| Where the mlp's gap to a bare sweep goes (answered: 12% is the row block's access pattern and 19% is three inner-loop uops, one of which GFNI already removes) | 0.9.5 |
| Whether the batched path's per-lane epilogue or its staging stride explains the marginal lane (answered: neither — 4 to 9% and 7%, against a gap of four times) | 0.9.5 |
| A fresh profile of a picture, part by part, which the vision entry opened by asking for (answered: scoring and the blend are 52.7% of a tower, and neither is waiting on memory) | 0.9.6 |
| A block of queries through the tower's attention, sharing the gathered run both loops read — at the one lane path's arithmetic in its order, held to it bit for bit by two tests | 0.9.6 |
| Whether flash attention's tiling is worth porting into this tower (refused: a band holds one score row and never held the grid, so the schedule arrives with nothing to save) | 0.9.6 |
| The fused multiply-add in the tower's blend (refused, and deliberately: 0.6x more, at the cost of a picture's rows no longer being the rows that ship) | 0.9.6 |
| A picture's rows kept against the picture, on a hundred and twenty-eight bit identity of the decoded samples and the tower's configuration | 0.9.7 |
| Where that identity has to be taken, so the decoder and the container stay out of it (answered: on the decoded raster, which makes a lossless re-encode one entry and a lossy one two) | 0.9.7 |
| Speculative decoding under a temperature (answered: the scout's proposal is a point mass, which is a `q` like any other — accept with `p(t)`, resample from `p` with the guess removed) | 0.9.8 |
| The sampler's shaped distribution, lifted out of the draw so two paths sit on one definition of what a taste means | 0.9.8 |
| The batched path's per-lane epilogue, which 0.9.5 measured and left (taken: the row's totals off the destination, prefill 4.4% and the marginal lane a tenth) | 0.9.9 |
| A patch budget the caller states, honestly plumbed — the resize moved and nothing else, and the budget inside the picture store's identity so two budgets cannot collide on one entry | 0.9.10 |
| What a patch budget costs, split by reading and by recognising as the entry demanded (answered: a six line notice survives to 35 rows, a four object scene to 12, and below six rows the model reports no picture at all) | 0.9.10 |
| Whether the vision tower is quadratic in patches at the grids it actually runs (answered: 3.25 ms a patch plus a term worth 17.8% at 2340, measured by skipping the tower rather than by subtracting a text turn) | 0.9.10, corrected in 0.9.11 |
| A picture's rows beyond one process — the backend and an encoder version in the mark, the path and the eviction left to the caller, and a refused file leaving the store alone | 0.9.11 |
| What half of a picture actually is (answered: not the tower — 9.25 s of 19.60 is the encoder and 9.46 is the 260 soft tokens through the text stack at a flat 36.5 ms each) | 0.9.11 |
