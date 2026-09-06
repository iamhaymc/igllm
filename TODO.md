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

A third is off this list because it is done. Reading a cached key row once for
all eight heads arrives in 0.8.5. It took decode on the byte cache from 5.91 to 7.22 tokens a second
and prefill from 14.85 to 18.94, moved no bit of any result, and was worth 5%
of decode on the float cache too, which was not the reason for it. `--cache 8`
is within a percent or two of the float cache on both builds now rather than a
third behind on one of them.

## Anywhere

- Read progressive jpeg. Baseline and extended sequential arrive in 0.8.5;
  `SOF2` is refused there the way the png reader refuses interlacing. It is a
  second decoder rather than a branch of the first: the coefficients arrive
  across several scans with successive approximation, so the whole picture's
  coefficients have to be held until the last scan lands, where a sequential
  block is final when its scan has read it. The marker walk, the Huffman decode,
  the transform, the upsampling and the colour transform are all already there
  and unchanged by it; what is new is the coefficient store, the four scan kinds
  — dc first, dc refine, ac first, ac refine — and the end-of-band run.
  Progressive is a fair share of what a browser is served, so it is worth
  having; it is not what a caller with a photograph on disk usually has.

  Two smaller gaps beside it: a four component file, CMYK or YCCK, needs the
  Adobe `APP14` transform flag read and an inversion rule for the ones written
  inverted, and neither can be guessed from the pixels; and a precision of
  twelve bits, which the extended sequential frame header allows, needs the
  tables and the level shift widened.
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
- Persist and restore a conversation, not only a prompt. 0.8.5 gives
  `session_save` and `session_load` and the `--keep` flag over them, and what
  they hold is a prompt: the file is written before the first token is sampled,
  so a rerun of the same prompt starts where the last run started. A file
  written that way cannot be picked up as a conversation, because the session
  that wrote it is one id short of the prompt — the id `session_step` was about
  to be fed — and a turn framed onto it would drop that id. What resuming a
  conversation wants is a save taken after an answer, and a way to say which of
  the two a file is; the loop's `/save` and `/open` are the shape of it.
- Close what is left of the gap at the odd bit widths. 0.8.5 took three, five,
  six and seven bits off the bit stream walk and onto a block of eight codes,
  which is a byte boundary at every one of those widths: the fused dot is 6.5 to
  8.2 times what it was on AVX2 and about twice on SSE2, and `kern_code_spread`
  beside it 3.4 to 3.8. They are still a quarter of what two and four bits
  reach, and the reason is the block's word, which is assembled a byte at a time
  because the packing is defined by the bit stream rather than by the host's
  byte order. What would close it is an unpack in the shape the two bit path
  has — a shuffle over a wider load, thirty-two codes at a time — and a way to
  read the word wide that does not assume the host is little-endian, or a stated
  decision that it is. Synthetic weights reach every width, which the shipped
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

Four items are off this list because 0.8.5 did them: baseline jpeg, the wider
range of png, the multi-turn chat loop with several conversations on one model,
and the session cache written out and read back. The png reader now takes grey
at one, two and four bits, palette at one to eight, and interlaced files of
every kind, so there is no legal pairing of depth and colour kind left that it
refuses; what jpeg still refuses is a decoder of its own and is carried above,
and so are the two halves of the other three that 0.8.5 did not reach.

A fifth is closed rather than carried. How `app_diff.py` should treat a picture
the two sides decode differently — the question the jpeg item raised — turns out
to be answered already: `diff_tower` feeds the reference the patches the engine
says it read, out of the activation dump, and the seam's graph half is fed the
same, so the only thing either half reads from the picture file is its width and
height. The decoder is not in the comparison, and a jpeg case is held to the
same floor as a png one.
