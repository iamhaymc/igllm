# TODO

Open development tasks, most consequential first.

- Measure numerical parity against the transformers reference on a real
  checkpoint. The development sandbox could not reach `huggingface.co`, so
  every tensor name, group size, and tokenizer setting is taken from the
  reference source rather than from the files themselves.
- Confirm the weight tensor names against the shipped checkpoint and widen
  `model_prefix_pick` if the export uses a prefix the loader does not expect.
- Confirm the chat frame against the `chat_template` in the checkpoint rather
  than the assumed `<start_of_turn>user … <end_of_turn>` shape.
- Support mixture-of-experts blocks. `config_read` currently refuses a
  configuration with `enable_moe_block` set.
- Support the vision and audio towers. The engine is text-only today.
- Batch the prefill. Prompt tokens are processed one at a time, so prefill
  runs at decode cost per token; a batched path would turn the projections
  into matrix products.
- Add vector paths for `kern_dot_code`. The packed dot product is scalar on
  every target; the two and four bit cases are the ones worth widening.
- Add an AVX-512 path beside AVX2, selected by the same macro layer.
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
- Record end-to-end logit fixtures once a checkpoint is available, and assert
  against them in `app_test.c`.
- Verify the Windows build on a real Windows host. The platform shims compile
  under cross-inspection only.
