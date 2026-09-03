# TODO

Open development tasks, most consequential first.

- Support the vision and audio towers, checking each piece against the
  synthetic oracle as it is written rather than trusting it in bulk. The
  engine is text-only today. The
  reference shape is a 16-layer bidirectional vision encoder with 2-D rotary
  positions and a 3x3 average pooler, and a 12-layer audio encoder with a
  convolutional subsampler and relative-position attention, each followed by
  a projection into the text embedding space. It also needs an image reader,
  a bicubic resize, a WAV reader, and a mel filterbank.
- Extend `app_fake.py` with vision and audio presets, so the towers have a
  reference to be checked against before any of them is written.
- Record end-to-end logit fixtures from the shipped checkpoint and assert
  against them in `app_test.c`, so the parity result survives in a build that
  has no checkpoint and no python.
- Profile and speed up decode. Parity is settled; throughput is not.
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
- Run the sanitizers over the quantization changes. `run.py test --debug` needs
  a toolchain that ships `libasan` and `libubsan`, which the MinGW build used
  for the Windows run does not.
