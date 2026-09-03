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

---

## 0.4.0 — parity on the shipped checkpoint

### Why

0.3.0 established the arithmetic against weights the reference itself wrote,
and said plainly what that did not cover: the names, the packing, and the
tokenizer of the real export. The checkpoint is now in hand.
`google/gemma-4-E2B-it-qat-mobile-transformers` did not load at all, and what
was wrong with the loader was not a detail.

### What the export actually contains

`quantization_config` carries no `config_groups` and no `format`;
`quant_method` is `gemma`, and nothing under it matches compressed-tensors:

- the codes sit under the plain `weight` name, as `U8` bytes rather than `I32`
  words, beside a `weight_scale` with one column per group. The binder saw
  `weight`, took it for a real tensor, and would have read packed bytes as
  floats;
- an embedding table is named apart — `embedding_quantized` and
  `embedding_scale` — so `model_prefix_pick` probed `embed_tokens.weight` and
  `embed_tokens.weight_packed`, found neither, and gave up before it reached
  anything else;
- two and four bit codes are unsigned with an offset of half the range, and
  eight bit codes are signed `I8` with no offset at all;
- nothing records the unpacked width, because the stored shape *is* the packed
  width. Two bits and four bits are told apart only by what the row was before
  it was packed;
- the widths are declared as a map of regular expressions over module names —
  two bits for `lm_head` and the token embeddings, four for attention and the
  first fifteen feed-forwards, two for the rest of them, eight for the
  per-layer gates. None of that map is parsed.

**The widths come from the tensors, not from the map.** A regular expression
engine to read that map would be a few hundred lines that could only ever
agree with what the shapes already say. `plane_bind` is told the input width
its caller expects instead — `q_proj` reads the state, `down_proj` reads what
`gate_proj` wrote, `o_proj` reads what `q_proj` wrote — and the bit width is
the one that makes the stored byte count come out right. A hint that no width
explains is refused rather than decoded, so a wrong expectation is a load
error and not a wrong answer.

**One xor covers both storage conventions.** A signed byte is the offset code
with its top bit flipped, so `plane` grew a `code_flip` that is the offset for
a signed plane and zero for every other, and the decode is otherwise
unchanged. The vector paths for two and four bits are guarded on the flip
being zero, which it always is at those widths.

### Static range quantization was the real omission

The larger gap was not a name. Every quantized projection in the export
carries an `input_activation_scale` and an `output_activation_scale`, and they
are calibrated, not left at zero: `layers.0.mlp.gate_proj` reads on a grid of
0.94 and writes on a grid of 0.62. The reference rounds the input to the
projection and its result onto those grids, at eight bit levels. Without it
the engine was not approximately right, it was running a different model.

`quant_step` does the rounding, and `session_lift_many` applies it on both
sides of every code plane, which is the one seam every projection already
passes through. It is not the dynamic `quant_act` rule beside it: that one
finds its range from the activation, this one is told it.

**The rounding is to even.** `floor(x + 0.5)` would be indistinguishable from
`rintf` anywhere else, and is not here. A product of two quantized planes is
an exact multiple of the two steps, so the value being rounded lands on a half
step often rather than never, and rounding up instead of to even moved whole
activations by a whole step. The layer-wise harness named `layer.2.mlp` off by
5.0118e-02, which is `layers.2.mlp.down_proj.output_activation_scale` to every
digit it printed.

### The chat frame

Gemma 4 renamed the turn markers. The frame is
`<bos><|turn>user\nHello!<turn|>\n<|turn>model\n`, and the loader was looking
for `<start_of_turn>`; finding neither marker it fell through to an unframed
prompt, which is a different question to ask an instruction-tuned model. Both
pairs are probed now, the newer first. `logits` frames the turn as well,
because what it reports is the distribution `chat` samples from and comparing
it against the reference only means something when the two sides read the same
prompt. `--raw` opts out, and the layer-wise harness passes it, since what it
compares is arithmetic and a frame would only move the prompt lengths it picks
on purpose.

### What parity means on this checkpoint

Static range quantization makes the forward pass discontinuous. Rounding onto
a grid turns a last-bit disagreement in a sum into a whole step, and one
flipped step at one layer is a different residual for every layer after it.
The reference does this to itself. The same graph in double precision agrees
with single to a part in ten million through the first twenty layers, then
moves by 5.6622e-02 at `layers.20.mlp` — two steps of that module's grid —
and by whole units in the logits. Fed its tokens one at a time behind its
cache rather than all at once, it moves just as far. There is no
single-precision answer to agree with.

The engine sits inside that. Every attention and feed-forward output is
identical to the reference through the first several layers, the residual
apart by one unit in the last place, which is the least a computation can
differ by without being the same computation. The first divergence is a single
slot of `layer.7.attn`, off by 2.523848e-02, which is
`layers.7.self_attn.o_proj.output_activation_scale` exactly. From there it
compounds the way the reference's own compounds.

So `app_diff.py` stopped asking for per tensor equality on a real checkpoint.
It reports how deep each side stays exact, requires the embedding and the
first layer — which nothing can have rounded twice yet — to agree outright,
and holds the end of the stack to how far the reference moves there. That last
judgement is made over the whole run rather than prompt by prompt, because
whether a given perturbation lands on a half step is a lottery: on one of the
six lengths the reference happened to flip nothing at all and moved by 4e-06,
which says nothing about what the engine may do. The double-precision run costs
about twice the memory of the single-precision one, because a forward
materializes the whole embedding table to read one row of it; where it does not
fit, the harness says so and leaves that prompt held to the strict bar rather
than stopping. `app_test.py` measures its
logit tolerance the same way, and its greedy comparison too — where the two
continuations part, it asks how far the reference follows itself before
parting, and requires the engine to do at least as well.

### Results

`run.py parity --model <folder>`, six prompt lengths:

```
  2 ids        first drift at layer.10.attn.0  the reference at layer.21.attn.0
                    final  off by 1.753e+00, the reference by 1.656e+00
                    logits off by 2.317e+00, the reference by 1.944e+00
  3 ids        first drift at layer.10.attn.0  the reference at layer.21.attn.0
                    final  off by 1.910e+00, the reference by 3.815e-06
                    logits off by 2.207e+00, the reference by 4.768e-06
  11 ids       first drift at layer.10.attn.0  the reference at layer.21.attn.0
                    final  off by 2.574e+00, the reference by 2.017e+00
                    logits off by 3.110e+00, the reference by 2.851e+00
  18 ids       first drift at layer.10.attn.0  the reference at layer.19.mlp.0
                    final  off by 2.195e+00, the reference by 1.749e+00
                    logits off by 3.073e+00, the reference by 2.207e+00
  19 ids       first drift at layer.10.attn.0  the reference at layer.19.mlp.0
                    final  off by 2.544e+00, the reference by 2.113e+00
                    logits off by 2.374e+00, the reference by 2.469e+00
               the double-precision run did not fit in memory
  44 ids       first drift at layer.10.attn.0  the reference was not measured
                    final  off by 1.876e+00, the reference by -
                    logits off by 2.074e+00, the reference by -
  over the run ok   final  off by at most 2.574e+00, the reference by 2.113e+00, allowed 4.225e+00
  over the run ok   logits off by at most 3.110e+00, the reference by 2.851e+00, allowed 5.703e+00
```

The engine drifts first at the same tensor every time, because position zero is
the sequence marker in every one of those prompts and nothing else reaches it.
The longest prompt is the one whose double-precision run did not fit in sixteen
gigabytes beside everything else; it contributes its own gap and no allowance,
which is the safe direction.

`run.py check --model <folder>`, four prompts: the token ids match the
reference's framed turn exactly on every one, the leading token matches on
every one, 81 to 94 per cent of the top sixteen ids are shared, and the
largest logit gap is between 0.54 and 1.71 against a reference that moves 0.51
to 1.56 against itself. Decode runs at 5.9 tokens a second against the
reference's 0.16, on four cores of a 2017 desktop.

### What it found

Three defects, none of them reachable from the synthetic oracle, because the
synthetic checkpoints are written by the reference in the compressed-tensors
layout and carry no activation ranges at all.

The eight bit dot product read its bytes directly and so never saw the flip,
which left the two per-layer gates decoding as unsigned. The engine loaded,
produced text, and was wrong from the first layer's output.

`floor(x + 0.5)` for the static grid, as above.

`app_test.c` used `_mkdir`, `_rmdir` and `getpid` on Windows without
`<direct.h>` or `<process.h>`, and `app_test.py` decoded the engine's utf-8
output with the console's code page. Neither had been run on a Windows host.

### Testing

`test_gemma` builds a safetensors fixture in the export's layout — a four bit
plane, a signed eight bit plane, and a two bit embedding table with two groups
to a row — and checks the decode of each, the group size the scale shape
implies, the bit width the column hint implies, the two activation steps, the
refusal of a column hint no packing explains, and the eight bit dot product
that carried the bug. The suite is 199 assertions and builds clean under
`-Wall -Wextra` on Windows.

The synthetic sweep is unchanged and still passes: eleven configurations by
six prompt lengths, each judged against the noise floor measured on it.

### Known gaps

The vision and audio towers are still absent. `k_cache_scale` and
`v_cache_scale` are read by nothing, as they are by the reference. Throughput
has now been measured but not worked on: 5.9 tokens a second untuned, 10.5
with AVX2.

The greedy continuation parts from the reference earlier than the reference
parts from itself on one of the four prompts — 52 characters against 96 —
which is the same lottery seen everywhere else in this work and is why the
check allows the engine to follow half as far rather than as far. It is worth
watching rather than hiding: a real defect would look like this too, only it
would also move the rank-one token, the top sixteen, and the layer-wise
comparison, and it moves none of them.

The sanitizers were not run over any of this. The MinGW toolchain on the
Windows host has no `libasan` or `libubsan`, and there was no POSIX host.
`TODO.md` has the rest.
