# CHANGES

Development progress and the reasoning behind each decision.

---

## 0.1.0 — initial engine

### Scope

A dependency-free C11 inference engine for Gemma 4 E2B IT QAT that loads a
Hugging Face checkpoint directly and runs it on the CPU, with the structure
in place for an accelerator backend later.

### Layout

**One translation unit.** `app_core.c` holds the whole engine; `app_main.c`
and `app_test.c` include it. The alternative — a header plus an object file —
would have needed a second file that the agreed file list does not allow, and
would have cost the compiler its whole-program view. The public interface is
still separated, inside `APP_CORE_INCLUDED`, so the API a caller sees is
exactly the API that is documented.

**Ten layers, bottom up.** Platform, JSON, store, quant, kernel, backend,
model, token, session, and the public bodies. Each layer depends only on the
ones beneath it, which is what makes the file readable at four thousand lines.

### Weights

**Zero copy.** The compressed-tensors pack-quantized layout turned out to be
a dense little-endian bit stream per row: element *i* at bits
`[i*bits, i*bits+bits)`, rows padded to a 32-bit boundary. The reference
packs in chunks of 32 bits, but that is algebraically the same stream, so on
a little-endian host the mapped bytes are directly usable. The engine
therefore keeps no second copy of the weights and pays no repacking pass at
load. Startup is the cost of parsing two JSON files.

**Scales stay in their stored dtype.** Converting every group scale to `f32`
at load would have cost roughly a quarter of a gigabyte. Converting per group,
inside the matrix product, costs one conversion per group per row — far below
the cost of the group itself.

**Bit width is measured, not trusted.** `plane_bits_of` recovers the width
from the logical shape and the packed word count. This checkpoint mixes
widths — the embedding tables are narrower than the projection matrices — and
a single configuration value cannot describe that.

**Shapes are derived from tensors.** Head size comes from the row count of
`q_proj`, key-value head count from the row count of `k_proj`. Configurations
and weights disagree often enough that trusting the weights is the safer rule.

### Arithmetic

**The matrix product is rearranged.** The obvious form subtracts the offset
and the zero point from every code. Factoring instead as

```
row = Σ_groups gain * ( Σ_j a[j]*code[j] - (offset+zero) * Σ_j a[j] )
```

removes a subtraction per element and lets the per-group activation sums be
computed once per product and shared by every row. It is exactly equivalent,
and it accumulates in a wider range.

**Rotary angles are evaluated, not tabulated.** A table for the model's
maximum position count would be hundreds of megabytes. Two transcendental
calls per pair per token is not measurable beside the projections.

**Softmax streams.** Maximum then sum, both in one pass over the row, so a
long attention row cannot overflow.

**RMSNorm uses the weight directly.** Gemma 4 follows Gemma 3n here, not
Gemma 2 or 3 — there is no `1 + weight` term. Getting this wrong produces
output that looks plausible and is wrong, so it has its own unit test.

**Attention scaling is one.** The query and key norms absorb the usual
`1/sqrt(d)`. This matches the reference and is easy to "fix" incorrectly.

### Memory

**Caches are sized per layer.** A sliding layer needs only its window and
uses a ring. A full-attention layer, and any layer that is the source for a
sharing layer, keeps the full context. Sharing layers alias their source and
allocate nothing. Sizing every layer for the full context would have cost
close to a gigabyte for what the model actually needs in tens of megabytes.

**Sessions allocate once.** Every scratch buffer is sized from the model at
`session_open`. The token loop allocates nothing, so the steady state has no
allocator in it and no fragmentation over a long conversation.

### Threading

**Fork and join, no queue.** `pool_run` gives every worker the same function
with a slice index, waits, and returns. A work-stealing queue would help an
irregular workload; a matrix product split into equal bands is not one, and
the queue would only add per-task synchronization to the hot path.
`slice_span` spreads the remainder across the leading bands so no worker
trails by more than one row.

### Accelerator readiness

**One table, bound once.** Every kernel is reached through `back_desk`. The
model and session layers never name a kernel. The loader keeps the bytes of a
weight separate from the handle to a weight. Those two properties together
are what let a device backend be added without editing the model graph.

### Tokenizer

**Driven entirely by `tokenizer.json`.** Metaspace behaviour is read from the
normalizer and pre-tokenizer records rather than assumed, byte fallback is
indexed from the `<0xNN>` entries, and control tokens are marked from
`added_tokens` so decoding can drop them.

### Testing

**Independent references, not recorded outputs.** Each kernel is checked
against a plain scalar definition written separately in the test file, and
`pack_read` is checked against a bit writer that shares no code with the
reader. A test that replays a recorded output only proves the code has not
changed; these prove it is right. The suite runs clean under the address and
behaviour sanitizers.

### Known gaps

The vision and audio towers are absent. Most
importantly, numerical parity has not been measured, because the checkpoint
was unreachable from the development sandbox — every name and layout here
comes from the reference source, not from the files. `TODO.md` tracks all of
this.

---

## 0.1.1 — mixture blocks, batched prefill, wider packed dot

### Mixture-of-experts

**The block is a second branch, not a replacement.** The reference runs the
dense mlp as a shared expert and adds a routed branch beside it, each under
its own post-norm, summed under a third. `session_layer` follows that shape
exactly rather than the more familiar "either dense or routed" form, because
the checkpoint carries all three norms and dropping any of them would be a
silent numeric error.

**The router reads the pre-mlp residual.** Not the normalized activation the
shared expert sees. Softmax over the experts, top *k*, renormalize the kept
weights to sum to one, then scale each by its `per_expert_scale`.

**Stacked expert weights are sliced, not copied.** `gate_up_proj` arrives as
one `[experts, 2*inner, hidden]` parameter with no `.weight` suffix.
`plane_bind_part` drops the leading axis by offsetting the payload, the
scales, and the zero points, so an expert is a view and the engine still
keeps no second copy of the weights.

**The mixture branch stays one lane wide.** Every token picks its own
experts, so batching it would mean either running all experts for the batch
or gathering per expert. Neither pays at the batch sizes prefill uses.

### Batched prefill

**Lanes, not a separate path.** Every session scratch buffer holds
`KERN_LANE_LIMIT` lanes with a named stride, and the whole graph takes a lane
count. Decode is a batch of one, so there is exactly one code path and the
batched arithmetic is exercised by every token the engine produces.

**Codes are unpacked once per batch.** `kern_row_code_many` spreads a group
of packed codes into a small float scratch and dots it against every lane, so
a batch pays the decode cost of a single vector. That is where the projection
becomes a matrix product: the weight traffic is amortized across the batch
rather than repeated per token.

**Keys are stored and read lane by lane.** A sliding layer's cache is a ring
narrower than the batch, so projecting the whole batch and then writing every
key would overwrite slots a lane still needs. The projections are batched;
the cache write and the attention read stay in position order.

### Wider packed dot

**Two and four bits are widened; three and five are not.** A four-bit code is
a nibble and a two-bit code is a quarter byte, so both unpack with shifts and
masks over whole vectors. The odd widths straddle byte boundaries and would
cost more in shuffles than the scalar loop costs outright.

**Alignment is checked, not assumed.** The vector paths need the group to
start on a byte boundary — even for four bits, a multiple of four for two —
and fall through to the bit-stream loop when it does not. The previous scalar
code silently assumed both, and dropped the tail of a group whose span was
not a multiple of the step.

### Testing

**The whole graph now runs in the suite.** `test_wing` writes a complete
miniature checkpoint — config, tokenizer, and a safetensors file with every
tensor the loader binds, dense and mixture — then asserts that a batched
prefill lands on exactly the logits produced by feeding the same tokens one
at a time. That is what caught the ring-buffer ordering bug above.

### Known gaps

The vision and audio towers are still absent. Numerical parity against the
reference has still not been measured, because the checkpoint remains
unreachable from the development sandbox. `TODO.md` tracks the rest.

---

## 0.3.0 — a synthetic parity oracle

### Why

Everything above was argued from the reference source rather than measured
against it, because the checkpoint could not be downloaded. But parity does
not actually need the shipped weights — it needs *some* weights both engines
agree on. `app_fake.py` makes them, and the download stops being a blocker
for the arithmetic. It remains a blocker for the file naming and the real
tokenizer, and `README.md` now says exactly that instead of disclaiming
everything at once.

### The checkpoint comes from the reference, not from a reading of it

`app_fake.py` builds a `Gemma4Config`, instantiates the reference model, and
lets `save_pretrained` write `config.json` and `model.safetensors`. The
alternative — writing the fixture by hand, as `test_wing` does — produces a
file that agrees with the loader's assumptions by construction, and so cannot
contradict them. A generated file can.

**Every parameter is reseeded.** The reference initializer sets norm weights
to exactly 1.0, and 1.0 makes a misindexed norm indistinguishable from a
correct one. `form_check` then rejects any model with a parameter left near
zero. That guard was not speculative: `post_feedforward_layernorm_2` does not
end in `norm.weight`, an early name-based test missed it, and the mixture
branch it gates came out at 4e-7 — small enough that an engine emitting
zeros would have passed. The test is on the module type now.

**The packing is upstream's.** `weight_packed` is produced by
`compressed_tensors.pack_to_int32` rather than by a second implementation of
the bitstream, so agreement means agreement with the exporter. A dequantized
twin is written alongside, and the engine reading the packed files is
compared against the reference reading the dequantized ones, which separates
the unpacking and the scale convention from the arithmetic.

### Comparison is layer-wise

A logit gap says something is wrong and nothing about where. `APP_TRACE`
compiles in a dump of named activations — embedding, per-layer attention,
feed-forward or mixture, residual, final norm, logits — and `app_diff.py`
compares them in order against forward hooks on the reference, stopping at
the first tensor that drifts. The dump is behind a macro and compiles to
nothing by default, so a normal build pays nothing for it.

**The tolerance is measured.** Running the reference over a whole sequence
and then one token at a time behind its cache is the same arithmetic in a
different summation order; the largest gap between them, 1.55e-6, is the
floor below which a difference means nothing. Tolerance is eight times it.

**The oracle was tested by breaking the engine.** Perturbing a single expert
weight makes the harness name that layer and stay quiet about the earlier
ones. Without that, a harness that always passes looks exactly like a correct
engine.

### What it found

Two real bugs, neither reachable from the unit tests.

`config_read` defaulted `global_head_dim` to 512. A `save_pretrained` config
never contains that key — the reference pops it in `__post_init__` and writes
`per_layer_config` instead — so the full-attention rotary table was built for
head dimension 512 while the weights had 16, and `rope_wave` wrote past its
buffer. `test_wing` missed it because a hand-built config includes the key.
The fix reads `per_layer_config`, and then `model_bind` rebuilds the tables
from the head size implied by `q_proj`, because the tensors are the only
source that cannot disagree with the weights.

`run.py` used `argparse.REMAINDER` for its trailing arguments, which
swallowed its own flags: `--debug` and `--trace` parsed successfully and did
nothing. That explains an earlier session's note that the sanitizer build
"did not appear to change the flags" — it did not. The argument vector is
split on the first bare `--` now.

The sweep also retroactively confirmed the mixture blocks and the batched
prefill from 0.2.0, which had been argued rather than measured.

### Coverage

Eleven configurations by six prompt lengths: dense and mixture, float and
packed at two, three, four, five and eight bits, group sizes that divide the
row and group sizes that do not, the double-wide feed-forward, shared key and
value projections, and lengths that straddle both the sliding window and the
prefill chunk. Clean under the sanitizers and on AVX2 and NEON.

### Known gaps

The vision and audio towers are still absent — but they are now buildable
against something, which is the point of the exercise. The tensor naming of
the shipped export, the real tokenizer's merges, real vocabulary edge cases,
and speed at full scale still need the checkpoint.
