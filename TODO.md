# TODO

Open development tasks. The first group can only be judged where the shipped
export is — against the real weights, or against the export's own configuration
read by the reference's own code — so it belongs on a host that has `model/`
beside it. The second group can be written and judged anywhere, on synthetic
weights or on none. Most consequential first within each group.

## On the shipped export

- Spend fewer instructions in the decode kernels, further. 0.8.4 took the two
  bit path from two broadcasts a sixteen codes to one and got 21% of decode at
  one thread, 9% at four; the narrowing is the shape of a build walking towards
  the memory. It is not there yet: at four threads the tuned build reads 5.72
  GiB/s of the 27.24 a bare sweep gives on that host, so four fifths of the
  machine is still unused. What is left is the four bit path, which is 335 MiB
  of the 727.5 a step sweeps and did not answer to either candidate 0.8.4 tried,
  and the eight bit path at 26 MiB. Both want a reading of what they are
  actually waiting on rather than another guess.
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
- Speed up the towers further, where the time is. 0.8.4 profiled a picture at
  the full patch budget — a 48 by 48 grid, 2304 patches, sixteen layers of
  twelve heads — and of the 148 s a single threaded run takes, the scoring is
  9.6 s, the softmax 6.6, the blend 7.9 and the projection out of attention 2.6.
  The softmax is 0.8.6's: the kernel is a fifth of what it was on the tuned
  build and half of it on the default one, which took a picture from 122.8 s to
  113.2 on the host 0.8.6 measured. What is left of the twenty-four seconds is
  the scoring and the blend, which the compiler was already vectorizing, and
  what is left of the picture is the six hundred billion multiply-adds of the
  projections themselves, which already run through the packed kernels. So the
  next reading here is of those kernels rather than of the tower, which is the
  item at the head of this list — and it wants a fresh profile, because the one
  quoted above is 0.8.4's and the softmax line of it is gone.

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
- Give the odd widths a vector path for the spread. 0.8.6 read the block of
  eight codes as one word rather than assembling it a byte at a time, which took
  all four of three, five, six and seven bits onto the same rate — about 6.0 G
  codes a second on the tuned build against 3.1 to 4.5 before, and 1.6 on the
  default build against 1.2 to 1.4 — so the width no longer shapes the loop. The
  fused dot is 38% of what two and four bits reach rather than a fifth to a
  quarter, and the remaining distance is nearly all in `kern_code_spread`, which
  still writes the block to scratch and reads it back a value at a time: 2.5 G
  codes a second against 20 at two bits. What would close it is the unpack in
  the shape the two bit path has — a shuffle over a wider load, thirty-two codes
  at a time — which is a different shuffle per width and is why it was not taken
  with the word read. Synthetic weights reach every width, which the shipped
  export does not, so none of this moves a token on the checkpoint that ships.
- Add an AVX-512 path beside AVX2, selected by the same macro layer. On the
  evidence it would have something to show rather than nothing: the tuned build
  used 42% of what the memory gives on the 2017 desktop and 20% on the virtual
  machine 0.8.4 was measured on, so a wider kernel has room to move the token
  rate on either. Neither of those two hosts is the place to judge it — the
  desktop has no AVX-512 at all, and the virtual machine is far enough from its
  memory that a wider kernel would flatter itself there.
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
