# TODO

Open development tasks. The first group can only be judged where the shipped
export is — against the real weights, or against the export's own configuration
read by the reference's own code — so it belongs on a host that has `model/`
beside it. The second group can be written and judged anywhere, on synthetic
weights or on none. Most consequential first within each group.

`RESEARCH.md` is a separate list, written on the assumption that this one is
finished, and it is not a queue this file feeds into. Where it touches an entry
here the entry says so. What it changed about this file is one item, closed
below: the eight lane block's open question was whether to reassociate a sum for
16%, and the answer is that reassociation is worth spending somewhere else.

It also puts a ceiling on the first group that is worth stating once, because
every entry in it is a decode or prefill entry and none of them can pass it. A
token that costs one sweep of the weights cannot be made to cost less than the
sweep. On the fourth host — four cores of a Xeon at 2.8 GHz, AVX-512 and VNNI,
the wide build — a bare sweep gives 32.18 GiB/s at four threads and a step reads
784.4 MiB, so a token that spent nothing at all outside the memory would take
24 ms: **41 tokens a second, and no more.** Whether to go past that is the
question `RESEARCH.md` opens rather than one this file can answer.

0.8.9 changed where in that gap the engine sits, and so changed what this list
is for. Every kernel that reads a code plane is now at the memory — the three
widths stream 25.90, 29.47 and 31.04 GiB/s of codes at four threads against the
sweep's 32.18 — and decode runs at 12.57 tokens a second of the 41. So the
kernels are no longer where a token goes, and the first entry below is a
measurement rather than a kernel, because nothing here can name the next kernel
until that measurement exists.

## On the shipped export

- Measure what a token spends outside the kernels, a part at a time. This is
  what is left of the head item, and it is now the largest thing on this list.

  0.8.9 took `RESEARCH.md` idea 2 and the kernels are finished: at four threads
  the two bit path reads 25.90 GiB/s of codes, the four bit 29.47 and the eight
  bit 31.04, against a bare sweep of 32.18 on the same host. All three are at
  the memory. A quicker kernel buys nothing at any width now, and the ceiling
  the paragraph above states is what is left of the whole group.

  What that leaves is the distance to it. Decode at four threads is 12.57 tokens
  a second, reading 9.63 GiB/s of the 784.4 MiB a step sweeps; the memory would
  hand those bytes over in 24 ms, which is 41 tokens a second, and the step takes
  80. So more than two thirds of a token is now outside the kernels, where 0.8.8
  left a third — the attention, the norms, the sampler, the 277 forks and joins,
  and the bands that do not divide evenly, none of which has been measured a part
  at a time on any host. That measurement is the next thing to do, and until it
  exists nothing here can say what the next kernel should be, because there may
  not be one.

- The other widths of the same instruction. The integer path is written for
  AVX-512 VNNI and nothing else. A host with `avx_vnni` and no AVX-512 —
  everything from Alder Lake on — has the same instruction at 256 bits, and an
  ARM host has `sdot`/`udot`, which is the same shape at 128. Both are the same
  kernel at another width and neither is written, for the reason 0.8.7 gave for
  AVX-512 itself: the path was written when a host with the instruction and the
  headroom to show it turned up, and these wait for the same.

  `RESEARCH.md` idea 3, the lookup table execution, belongs beside this rather
  than before it. It answers the same question the integer path answered — how
  to stop spreading codes into floats — so on a host that has an integer dot
  product there is nothing left for it to win. On a host that has none, it is
  the only route to the same place, and that is where the experiment now
  belongs.

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
- The tower's own attention, which is what a picture is now mostly made of.
  0.8.9 took the tower's projections with everything else — they are eight bit
  and they are a batch, which is the shape that gained most — and that has moved
  the balance inside a picture rather than only shortening it. On the fourth
  host, one thread, the same 768 by 768 picture at the full patch budget: the
  whole picture costs 48.3 s where it cost 105.0, of which prefilling the 256
  soft tokens through the text stack is 16.3 s where it was 50.7. What is left
  for the tower is about 32 s where it was about 54 — a third off, against the
  text stack's two thirds.

  That is the reading `RESEARCH.md` idea 11 was waiting for, and it now points
  at the tower's attention rather than at its projections: the scoring and the
  blend are float, were never touched, and are a larger share of a tower than
  they have ever been. Idea 11 is a schedule rather than a kernel — score a
  tile, carry the running normalizer, accumulate the blend, and never hold the
  whole score matrix.

  The two numbers above are a subtraction — the picture's wall clock against the
  same run without it, minus the prefill the tally reports — rather than the
  profile 0.8.8 took, which broke a tower into projections, scoring, blend and
  softmax. That profile should be taken again before the schedule is written,
  because the shares it recorded are the ones that just changed.

One item is closed rather than carried here. The eight lane block asked whether
16% of the batch loop was worth reassociating every sum in the engine — the
seventeenth live vector meant the quick form is the one with a single
accumulator a lane, at 30.13 G multiply-adds a second against the four lane
form's 27.15, and the wide tier's thirty-two registers held eight lanes and both
accumulators and measured 24.04, so the register count was never the obstacle
and there is no exact eight lane form to reach for. The answer is that it is
not. Once reassociation is on the table at all, the tower's attention above
wants it for a third of a tower and `RESEARCH.md` idea 2 wants it for an integer
accumulator that changes the arithmetic anyway, and both are larger than 16% of
one loop. The engine's last bit is only worth moving once, so it should be moved
for whichever of those two measures out, not here. 0.8.9 is that spending: the
integer accumulator is in, it moved every code plane's last bit, and it moved
the towers closer to the reference rather than merely elsewhere. The tower's
attention is the other claimant and is still open.

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
  accelerator backend can be dropped in without touching the loader. This is
  also the hook every kernel experiment in `RESEARCH.md` wants: an integer or
  table backend that packs a plane its own way at open needs somewhere to hold
  that packing, and `desk.mat_mat` behind a second `back_open` is that place. It
  was written here as an accelerator item and is a portability one first.
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
