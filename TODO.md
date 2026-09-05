# TODO

Open development tasks, most consequential first.

- Hold the seam open as the prompt grows. All four cases are judged on the
  shipped export now, each in a process of its own, and the longest is 316 ids
  with a picture and a clip in it. What has not been tried is a prompt long
  enough that the reference's forward stops fitting even alone — the cost is the
  reference's rather than the engine's, and the answer when it comes is probably
  to compare against a cached forward rather than a live one.
- Match the processor's audio token budget. The framing itself is settled: the
  engine's frame count is the live count `input_features_mask` marks, checked
  against the extractor on twenty-two clip lengths, and the rows it makes are
  `ceil(live / 4)` on all of them. What is not settled is the ceiling.
  `processor_config.json` records an `audio_seq_length` of 750 and an
  `audio_ms_per_token` of 40, which the reference pads or trims a clip to; the
  engine emits whatever the clip yields. Nothing has parted them yet, because
  every clip tested is inside the ceiling, but one past thirty seconds would.
- Speed up decode further. The batched path was the defect and is fixed —
  prefill is six times quicker on the default build and now beats decode per
  token — but decode itself gained only a third, because it runs one lane and
  its cost is the decode fused inside `kern_dot_code`. That loop no longer
  stalls on its own accumulator, and it still spends about two instructions a
  weight on the narrow widths. A byte-indexed table of unpacked floats would
  spend less: at two bits a byte is four codes, so a 256-entry table of four
  floats turns the unpacking into one aligned load.
- Speed up the towers further. Vision attention runs a head at a time over the
  pool, with each head's keys and values gathered into a run that stays in
  cache, and the eight bit projections the tower is quantized to now have a
  vector path: a whole run — an image at the full patch budget, and the 316 ids
  of prefill behind it — went from four minutes twenty-eight to one minute two
  on the SSE2 build. What is left: the score and blend loops are scalar where
  the dot product has a vector path, an image at the full patch budget still
  costs six hundred billion multiply-adds however it is arranged, and the
  conformer's convolution module still walks its kernel per channel per frame.
  None of it runs in the token loop.
- Support more than one image or clip per prompt, and interleave them with the
  text rather than putting them all in front of it. The engine substitutes any
  set of positions; it is the command line that assumes one of each.
- Read a wider range of pictures. The png reader refuses interlaced files and
  bit depths under eight, and there is no jpeg reader at all.
- Add an AVX-512 path beside AVX2, selected by the same macro layer.
- Widen `kern_dot_code` for the odd bit widths. Two, four and eight bits are
  vectorized, in the fused dot and in `kern_code_spread` beside it; three,
  five, six, and seven still walk the bit stream in both.
- Cache dequantized scales for the hottest planes. Gains are converted from
  their stored dtype on every group; a per-plane float mirror trades memory
  for a shorter inner loop.
- Let the caller choose weight residency per tensor rather than by size, so a
  memory-tight host can keep more planes packed.
- Add a device handle beside `plane` and a second `back_open`, so an
  accelerator backend can be dropped in without touching the loader.
- Support more than one concurrent session per model in the CLI, and add a
  multi-turn chat loop rather than a single turn.
- Persist and restore a session cache, so a long prompt need not be primed
  twice.
- Support the static ranges the export calibrates for the key and value cache.
  `k_cache_scale` and `v_cache_scale` are read by nothing — the reference
  ignores them too — but a backend that stores the cache quantized will want
  them.
- Build under MSVC. The suite builds clean and passes on a Windows host with
  MinGW gcc, on the scalar, SSE2 and AVX2 backends; `cl` and its `/arch:AVX2`
  path have still only been read.
