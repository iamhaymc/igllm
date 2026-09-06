# TODO

Open development tasks. The first group can only be judged where the shipped
export is — against the real weights, or against the export's own configuration
read by the reference's own code — so it belongs on a host that has `model/`
beside it. The second group can be written and judged anywhere, on synthetic
weights or on none. Most consequential first within each group.

## On the shipped export

- Spend fewer instructions in the two bit decode path, which is the one still
  spending them. 0.8.8 read what each width waits on at four threads, on rows
  streamed from memory against the same rows held in cache, and the three
  answers differ. The eight bit path keeps half of its resident rate and lands
  on the memory's own number — 23.95 GiB/s of the 26.27 a bare sweep gives on
  the tuned build, 27.66 on the wide one — so it is done, and a quicker kernel
  there buys nothing. The four bit path keeps 83% of its resident rate tuned and
  68% wide: the memory is part of what it waits on and there is a little left in
  it. The two bit path keeps 94% and 78%, and is 366 MiB of the 727.5 a step
  sweeps, which makes it the one width where instructions are still most of the
  cost and the only one worth another kernel.

  What that reading also says is that the kernels are no longer where a token
  goes at four threads. Streamed through the rates above, the mix a step sweeps
  is 38.9 ms of kernel a token, and the token is 63.7 ms on the wide build now
  that the fork and the join are 4.8 ms of it rather than 33. The 25 ms outside
  the kernels — the attention, the norms, the sampler, the bands that do not
  divide evenly — have never been measured a part at a time, and on this host
  they are now larger than anything left inside them.
- Hold the seam open as the prompt grows. All nine cases are judged on the
  shipped export now, each in a process of its own, and the longest is 664 ids
  with two pictures and a clip in it. What has not been tried is a prompt long
  enough that the reference's forward stops fitting even alone — the cost is the
  reference's rather than the engine's, and the answer when it comes is probably
  to compare against a cached forward rather than a live one.
- Let the caller choose residency per tensor. Two thirds of the 2334.8 MiB the
  export maps is never touched by a token — the per-layer embedding table at
  1120 MiB and the two towers at 324 — and what a step reads is 759.4 MiB. That
  is a footprint question rather than a decode one, as 0.8.1 established, and it
  is worth having on a small host. It is not a decode win and should not be
  scoped as one.
- Speed up the towers further, and the batch under them. 0.8.8 profiled a
  picture again — the tower is 39.1 s single threaded at the full patch budget,
  of which the projections are 24.0, the scoring 6.5, the blend 5.0 and the
  softmax 0.83 — and then took the one change that reading pointed at: four
  lanes of a batch now share the row's load rather than each loading it again,
  which is 8% of the projections and 23% of prefill on the default build.

  What is left of that loop is the eight lane block, and what stands in its way
  is not what it looks like. Eight lanes with two accumulators apiece is
  seventeen live vectors against sixteen, so the eight lane form that measures
  16% quicker — 30.13 G multiply-adds a second against 27.15 — is the one with a
  single accumulator a lane, which reassociates every sum in the engine. The
  wide tier has thirty-two vector registers and could hold eight lanes and both
  accumulators, and written that way it measures 24.04 against the four lane
  form's 27.15: slower, so the register count was never the obstacle. The open
  question is therefore whether 16% of this kernel is worth moving every logit,
  not how to fit the exact form into a host that has room for it.

  Past that the tower's own attention is what is left: the scoring at 6.5 s and
  the blend at 5.0, neither touched since they were written, both already
  vectorized by the compiler, and together a third of a tower.

- The other half of a picture is the text stack, and nothing has been asked of
  it. A picture at the full budget lays 256 soft tokens down, and prefilling
  those through the text stack costs 40.1 s single threaded against the tower's
  39.1 — half of what a picture costs, and outside every reading of a picture
  taken before 0.8.8. It is the ordinary prefill path, so the item above is most
  of what would move it, but it is worth stating that a tower made free would
  halve a picture rather than remove it.

Two items were closed rather than carried in 0.8.4. The per-group gain mirror — a
per-plane float copy of the group gains, traded against the conversions it
saves — has nothing to convert on this export: all 548 `weight_scale` tensors
are `F32` of shape `[rows, 1]`, so `plane_gain` is already a plain indexed float
load taken once per row, 1,203,456 of them a token against 727.5 MiB of codes.
The mirror would be a verbatim copy of 4.59 MiB for no conversion saved. A
checkpoint shipping `bf16` group scales would put it back on the table; this one
does not. And vectorizing the conformer's score loop was measured at nothing on
the shipped export — the window is thirteen keys wide — at the cost of moving
the tower's logits in the third decimal, so it was not taken.

Two more are off this list because they are done. Reading a cached key row once
for all eight heads arrived in 0.8.5. And the picture attention softmax is a
series rather than a call in 0.8.6: within one unit in the last place of `expf`,
five times its speed on the tuned build, and worth 8% of a single threaded
picture. It moves the numbers, which is why the entry for it records that
reassociating the old summation moves them as far — a picture has 35 routed
layers behind it and amplifies a last bit either way.

## Anywhere

- Run two conversations at the same time rather than one after the other. 0.8.5
  gives the CLI several sessions on one model and a loop that takes turns in any
  of them, and they take those turns one at a time: the sessions are
  independent, but every kernel underneath them reaches the model's one
  `pool_group`, which is a fork and join with no queue in it and one caller's to
  be inside at a time. Two of them stepping at once would be two callers in that
  fork. What it wants is either a pool a session owns — which costs a thread a
  session and gives the host's cores to whoever asks first — or a queue in front
  of the one pool, which is the shape a server wants anyway and is the larger
  change. Neither is worth guessing at without a caller that needs it.
- Add a device handle beside `plane` and a second `back_open`, so an
  accelerator backend can be dropped in without touching the loader.
- Build under MSVC. The suite builds clean and passes on a Windows host with
  MinGW gcc, on the scalar, SSE2 and AVX2 backends; `cl` and its `/arch:AVX2`
  path have still only been read.
- Read a jpeg this reader still has no decoder for. 0.8.6 leaves lossless,
  differential, hierarchical and arithmetic coded frames refused, each being
  another entropy coder or another frame shape rather than another branch. None
  of them is what a caller with a photograph on disk has, and arithmetic coding
  in particular is what nothing in the wild produces, so this is a completeness
  item rather than a useful one.

Two items are off this list because 0.8.8 answered them, and the first two items
above are what is left of them. The second is the towers: the fresh profile that
entry asked for is in `CHANGES.md` 0.8.8, and it moved the question from the
tower to the batch every tower and every prefill shares.

The first is the head one. What the four bit and eight bit paths wait on at four threads is
answered — the eight bit path waits on the memory, the four bit one partly — and
the larger half of the answer was in neither. Between the kernels sat 277 forks
and joins a token, 120 microseconds apiece at four threads, a third of the
token; both sides of the pool now spin briefly before they sleep, and decode at
four threads went from 10.51 tokens a second to 15.69 on the wide build, 8.37 to
11.21 on the tuned one and 6.19 to 7.26 on the default one, without moving a bit
of any result. A pool with more threads than the host has cores does not spin,
which was measured as well and is the other half of that finding.

Two items are off this list because 0.8.7 did them, one from each group as it
stood.

The spread's vector path at the odd widths is there, and it took the fused dot
with it, because both readers of a block now decode it the same way. A block of
eight always begins on a byte boundary, so every block of a width picks the same
bytes at the same shifts and the whole decode is one shuffle over a sixteen byte
load, with the tables built once for a run. On the host 0.8.7 measured, a five
bit row of 12288 at one thread: the spread 9.08 G codes a second against 1.21,
within a tenth of the 10.31 at two bits and 10.76 at four, which is the distance
closed rather than narrowed. The fused dot is 6.45 against 3.36, which is 59% of
the two bit rate where 0.8.6 left it at 38%. The shipped export packs no odd
width, so none of it moves a token there.

And AVX-512 is in, as a tier above AVX2 rather than an alternative to it: a host
with the one has the other, so the macro layer sets both names and only the
kernels with something to gain from sixteen lanes are written twice. The
evidence the item wanted arrived with a host that has both the instructions and
the headroom — a bare sweep gives 33.99 GiB/s at four threads there and the
tuned build reads 5.25 — and it showed both what to take and what to leave. The
fused dot went wide at the three widths the export packs, 36% to 54% quicker,
and on the export that is decode 21% quicker at one thread and prefill 7%. At
four threads it is 6% and 3%, because the memory is more of the cost once four
threads pull on it, which is the reading the head item above asked for rather
than a disappointment. The spread was written wide as well, measured 29% quicker
on its own, left prefill 13% slower, and was taken out again.

Four items are off this list because 0.8.5 did them: baseline jpeg, the wider
range of png, the multi-turn chat loop with several conversations on one model,
and the session cache written out and read back.

Three more are off it because 0.8.6 did them. Progressive jpeg arrives with the
two smaller gaps that were named beside it — `APP14` read, so three bands can
say they are the picture's own and four are read as ink, and twelve bits a
sample — so there is no jpeg this reader turns away now but one that wants
another entropy coder, which is carried above. A session file says whether it
holds a prompt or a conversation, and the loop's `/save` and `/open` put a
conversation down and pick it up in a later process. And the odd bit widths read
their block as one word, with the little-endian host stated once rather than
worked around in a kernel; what is left of that one is carried above as the
spread's vector path.

A fifth from 0.8.5 is closed rather than carried. How `app_diff.py` should treat
a picture the two sides decode differently — the question the jpeg item raised —
turns out to be answered already: `diff_tower` feeds the reference the patches
the engine says it read, out of the activation dump, and the seam's graph half
is fed the same, so the only thing either half reads from the picture file is
its width and height. The decoder is not in the comparison, and a jpeg case is
held to the same floor as a png one.
