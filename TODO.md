# TODO

Open development tasks. The first group can only be judged where the shipped
export is — against the real weights, or against the export's own configuration
read by the reference's own code — so it belongs on a host that has `model/`
beside it. The second group can be written and judged anywhere, on synthetic
weights or on none. Most consequential first within each group.

## On the shipped export

- Read a cached key row once for all eight heads. This export ships
  `num_key_value_heads` of one against eight attention heads, so `group_share`
  is eight and every head of a layer scores against the same cached key row and
  blends the same cached value row. `session_layer` decodes each of them once
  per head, which is eight reads of every byte where one would do. Blocking the
  loop the other way — a run of rows decoded once into scratch, then all eight
  heads over the floats — cuts the decode work by eight without giving back any
  of the storage the byte cache collects, and it serves every backend rather
  than only the one with the gather. 0.8.4 measured what that gather costs: the
  byte cache is 33% behind the float cache on the tuned build and 11% behind on
  the default, six pairs of one sign on a quiet host, and with it the tuned
  build is slower than the default build. This is the answer to that, and it is
  a restructuring of the attention loop rather than a kernel swap.
- Spend fewer instructions in the decode kernels, further. 0.8.4 took the two
  bit path from two broadcasts a sixteen codes to one and got 21% of decode at
  one thread, 9% at four; the narrowing is the shape of a build walking towards
  the memory. It is not there yet: at four threads the tuned build reads 5.33
  GiB/s of the 27.24 a bare sweep gives on that host, so four fifths of the
  machine is still unused. What is left is the four bit path, which is 335 MiB
  of the 727.5 a step sweeps and did not answer to either candidate 0.8.4 tried,
  and the eight bit path at 26 MiB. Both want a reading of what they are
  actually waiting on rather than another guess.
- Vectorize the softmax over a picture's attention scores. It is 6.6 s of the
  148 s a single threaded run of an image at the full patch budget takes —
  more than the blend and nearly as much as the scoring — and it is a scalar
  `expf` per patch pair per head per layer, a billion of them for one picture.
  It runs in the towers rather than in the token loop, so it is the cost of the
  first answer rather than of the run, and it is now the largest single thing
  in a picture that is not a matrix product.
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
  The rest is the projections and the feed-forward. So the loops 0.7.2 named as
  scalar are twenty-four seconds of a hundred and forty-eight, and the compiler
  had already been vectorizing them; 0.8.4 wrote the blend out through the macro
  layer anyway, which is worth 2% of a picture on the default build and nothing
  on the tuned one. What is left is the six hundred billion multiply-adds of the
  projections themselves, which already run through the packed kernels, and the
  softmax above.

Two items are closed rather than carried. The per-group gain mirror — a
per-plane float copy of the group gains, traded against the conversions it
saves — has nothing to convert on this export: all 548 `weight_scale` tensors
are `F32` of shape `[rows, 1]`, so `plane_gain` is already a plain indexed float
load taken once per row, 1,203,456 of them a token against 727.5 MiB of codes.
The mirror would be a verbatim copy of 4.59 MiB for no conversion saved. A
checkpoint shipping `bf16` group scales would put it back on the table; this one
does not. And vectorizing the conformer's score loop was measured at nothing on
the shipped export — the window is thirteen keys wide — at the cost of moving
the tower's logits in the third decimal, so it was not taken.

## Anywhere

- Read a wider range of pictures. The png reader refuses interlaced files and
  bit depths under eight, and there is no jpeg reader at all.
- Support more than one concurrent session per model in the CLI, and add a
  multi-turn chat loop rather than a single turn.
- Persist and restore a session cache, so a long prompt need not be primed
  twice.
- Widen `kern_dot_code` for the odd bit widths. Two, four and eight bits are
  vectorized, in the fused dot and in `kern_code_spread` beside it; three,
  five, six, and seven still walk the bit stream in both. Synthetic weights
  reach every width, which the shipped export does not.
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
