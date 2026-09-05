# TODO

Open development tasks, most consequential first.

- Verify the whole multi-modal graph end to end against the reference, not just
  each tower against its own. The language model of the shipped export
  dequantizes to about nineteen gigabytes, so `Gemma4ForConditionalGeneration`
  could not be loaded on the development host; the text stack and the two
  towers are each verified separately, and what is not yet checked is the seam
  — the placeholder run the processor lays down and the ids around it.
- Match the processor's soft-token policy. `processor_config.json` records
  `audio_seq_length` 750 and `audio_ms_per_token` 40, so the reference pads or
  trims a clip to a fixed token count; the engine emits whatever the clip
  yields, which is one frame short of the reference's own extractor on a clip
  that does not divide evenly.
- Record end-to-end logit fixtures from the shipped checkpoint and assert
  against them in `app_test.c`, so the parity result survives in a build that
  has no checkpoint and no python.
- Profile and speed up decode. Parity is settled; throughput is not.
- Speed up the towers. Vision attention is a plain triple loop over the whole
  patch grid, which at 2340 patches is the dominant cost of an image; the
  conformer's convolution module walks its kernel per channel per frame. None
  of it runs in the token loop, so none of it has been worked on.
- Support more than one image or clip per prompt, and interleave them with the
  text rather than putting them all in front of it. The engine substitutes any
  set of positions; it is the command line that assumes one of each.
- Read a wider range of pictures. The png reader refuses interlaced files and
  bit depths under eight, and there is no jpeg reader at all.
- Add an AVX-512 path beside AVX2, selected by the same macro layer.
- Widen `kern_dot_code` for the odd bit widths. Two and four bits are
  vectorized; three, five, six, and seven still walk the bit stream.
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
- Run the sanitizers over the tower and media changes. `run.py test --debug`
  needs a toolchain that ships `libasan` and `libubsan`, which the MinGW build
  used for the Windows run does not.
