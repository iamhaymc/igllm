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

---

## 0.5.0 — the vision and audio towers

### Why

The engine was text-only, and the two encoders were the largest thing missing
from an architecture that has them. They also bring in work the engine had
never done before: reading a picture, resizing it, reading a sound, and turning
it into a spectrogram — none of which can be borrowed in a project with no
dependencies.

### Both towers are the text layer with the mask taken off

`tower_feed` runs `input_layernorm` → attention → `post_attention_layernorm` →
residual, then `pre_feedforward_layernorm` → gated MLP →
`post_feedforward_layernorm` → residual. That is `session_layer` without the
per-layer embedding gate and without the mixture branch, so the two towers
share one copy of it, and what differs between them is exactly what should:
where a position enters the score, and how the input becomes a row.

Every tower projection goes through `plane_lift_many`, which is the function
the text stack's projections go through. That was a refactor rather than an
addition — `session_lift_many` used to hold the activation-grid rule and the
session's staging room together, and the towers needed the rule without the
session. Splitting them left the rule in one place, which is what the guide
already claimed, and it means a packed tower with calibrated activation ranges
needs no second code path. The staging room is chunked to `KERN_LANE_LIMIT`
lanes inside the function, so a tower running two hundred patches at once needs
no larger buffer than the token loop does.

### Positions

**Vision is two dimensional.** The first half of a head turns with the patch's
row and the second half with its column, which lets the rotate-half kernel the
text stack uses apply unchanged. It needs a head size divisible by four — each
axis takes half a head and is rotated as a pair — and a tower whose head does
not divide is refused rather than guessed at.

**Audio is relative.** The score is a content term against the key plus a term
against a projection of the distance between the two frames, each with its own
learned bias, which is what lets one query serve both. The relative rows do not
depend on the query, so they are projected once for the whole layer rather than
once for every pair of frames. Without `pos_proj` the tower is an ordinary
bidirectional encoder, which is what a checkpoint that left it out is asking
for.

### Where the rows go

A tower's output is substituted for the embedding of a placeholder token
**after** the token path has applied its `sqrt(hidden_size)` scale, not before
it, so a projector emits values already in the units the residual stream
carries. The id stays the placeholder's, because the per-layer embedding block
reads the id and the reference feeds it the placeholder too. `session_prime`
gained `session_prime_media`, which takes one embedding row per id and a mark
saying which of them a tower filled; the old entry points forward to it with
nulls, so nothing text-only changed.

### The media layer

**The deflate reader is here because a png cannot be read without one.** It
handles stored, fixed and dynamic blocks, skips a zlib wrapper when it finds
one, and refuses a truncated stream rather than returning what it managed —
the caller sizes its buffer from the header and would read the rest as image
data. The png reader undoes all five line filters and takes eight and sixteen
bit grey, RGB, palette and the two alpha forms; interlaced files and sub-byte
depths are refused. A pnm and a bmp reader come almost free beside it, and
`image_read` picks between the three from the leading bytes rather than the
file name, because a checkpoint's own sample images are as likely to be
misnamed as anything.

**The resize widens its filter when it shrinks.** A plain four-tap bicubic
sampling sixteen pixels out of a thousand aliases badly, and it is visible as
speckle in any downscaled photograph. The support is scaled by the reduction
factor, which is what PIL and torchvision do when antialiasing is on. Centres
are half-pixel, so the two images cover the same area rather than sharing a
corner.

**The filterbank is HTK mel.** Triangles at fractional bin positions, so two
neighbours hand over cleanly across the band they share; a periodic Hann
window; the transform taken over the next power of two at or above the frame;
and a floor under the logarithm, without which a silent band reaches negative
infinity and takes the tower with it.

### Testing

`test_puff`, `test_image`, `test_scale`, `test_wave` and `test_mel` hold every
piece of the media layer against something written independently of it. Two
fixtures were produced by a foreign encoder over content a formula describes —
four hundred bytes deflated into a dynamic Huffman block, and a twelve by eight
png with the five line filters used in turn — which is stronger than a recorded
output: a recording only shows that nothing has changed. The transform is
checked against the discrete sum it is a fast way of computing, and the
separable resize against a direct two dimensional gather.

`test_tower` writes a miniature multi-modal checkpoint and runs both towers
against a definition of them written out separately in the test file. The two
sides share the weights and the decoded media and nothing else. Finding the
audio reference wrong on its first run was the point of writing it: the fixture
carries query and key norms, the tower applies them, and the reference had left
them out.

`app_diff.py --media` does the same at a larger size and in another language,
against `tower_forward` in `app_fake.py`. Both oracles were then checked by
breaking the engine on purpose. Swapping the row and the column in
`rope_grid_wave` makes the harness report `vision.0.attn.0`, stay silent about
the patches and the embedding before it, and leave the audio tower passing;
reading the relative row for the opposite offset makes it report
`audio.0.attn.0` and leaves the vision tower passing.

The suite is 267 assertions and builds clean under `-Wall -Wextra`.

### What the tower oracle is worth

Less than the text one, and the difference is worth stating rather than
glossing. The text stack is compared against the reference library: the
checkpoint is written by `save_pretrained` and read by `AutoModelForCausalLM`,
so the names, the shapes and the arithmetic all come from upstream rather than
from a reading of upstream. No released `transformers` carries these towers, so
there is nothing upstream to compare against, and `app_fake.py` has to build
the tower weights itself.

What is left is two transcriptions of one description — the engine in C and
`tower_forward` in torch, plus a third in `app_test.c` — walked tensor by
tensor. That catches a transcription error, an index the wrong way round, a
norm in the wrong place, an offset read backwards; all of those are what the
deliberate perturbations above are. It cannot catch a misunderstanding the
sides share. `TODO.md` carries the item to replace it when upstream ships.

### Fixed on the way through

`token_decode_book` left its output slot untouched when it returned zero for a
control token, so a caller that reads the text without checking the count
printed uninitialized stack. Every media prompt starts with a run of
placeholder ids, which made it show up immediately. The slot is now terminated
before anything else happens.

### Known gaps

The towers have not been run against a real multi-modal checkpoint, because
none exists to run them against. Nothing in them has been optimized: attention
is a plain triple loop over the whole grid, the audio tower materializes a
relative row for every reachable offset whether or not the context spans reach
it, and the subsampler projects one frame at a time. None of that is in the
token loop.

The command line takes one picture and one clip and puts both in front of the
prompt. The engine substitutes any set of positions, so interleaving them with
the text is a front-end change rather than an engine one.

The png reader refuses interlaced files and bit depths under eight, and there
is no jpeg reader. The sanitizers were not run over any of this, for the same
reason as last time: the MinGW toolchain on the Windows host has no `libasan`
or `libubsan`, and there was no POSIX host. `TODO.md` has the rest.

---

## 0.6.0 — the towers against the reference

### Why

Version 0.5.0 shipped two towers that had never been compared to anything but
themselves. The reasoning given at the time was that no released `transformers`
carried the vision and audio encoders, so there was nothing upstream to compare
against, and that three transcriptions of one description agreeing with each
other was the best available evidence. That reasoning was correct about the
evidence and wrong about the premise: `transformers` **does** carry them, and
the shipped checkpoint is not gated. Installing the library from source and
downloading the export took twenty minutes.

The comparison then said what a comparison of that kind usually says when it is
run for the first time: almost everything was wrong.

### What the reference actually says

The architecture in 0.5.0 was in the right family and wrong in most specifics.

| | what 0.5.0 did | what the reference does |
| --- | --- | --- |
| vision rope | flat `rope_theta`, default 10000 | `rope_parameters.rope_theta`, **100** |
| vision rope pairing | rotate-half across the whole head | each half rotated **within itself** |
| vision axes | row first | **column** first |
| vision attention | scaled by `1/sqrt(d)` | scaled by **one**; the norms absorb it |
| vision values | no norm | a norm **without a scale** |
| vision positions | none, or one flat table | **two** tables, one per axis, summed |
| vision input | fixed square `image_size` | **variable resolution**, capped by a soft-token budget |
| patch layout | band, row, column | row, column, **band** |
| pooling | ragged windows, no scale | exact windows, scaled by `sqrt(hidden)` |
| audio layer | a transformer layer | a **conformer**: two half-weight feed-forwards and a gated depthwise convolution |
| audio activation | gelu | **silu** |
| audio attention | full or simple span | **chunked local**, with a tanh softcap and a per-channel query scale |
| mel spectrum | power, clamped log, derived transform length | **magnitude**, floor added under the log, stated length, semicausal padding |

Every one of those is now the reference's. The vision encoder layer's residual
arrangement was the one thing that had been right, which is why `tower_feed`
survived unchanged.

### The towers can be weighed even when the model cannot be loaded

The obvious way to check a tower is to load the model and compare. That was not
available: the shipped export's language model dequantizes to roughly nineteen
gigabytes, against eight free on the development host, and no amount of care
about dtypes closes that gap.

What made the comparison possible is that the reference gives each tower its own
`base_model_prefix`. A tower can be constructed on its own, handed the subset of
the checkpoint that belongs to it, and run — a couple of hundred megabytes
instead of nineteen gigabytes. `replace_with_quant_layers` then decodes the
packed weights, so the unpacking, the per-row scales and the activation rounding
on the reference side are all the reference's rather than a second reading of
them.

That is worth keeping in mind more generally: when a model is too large to hold,
the piece under suspicion often is not.

### What the comparison found

**The vision tower was right after one correction.** The rope pairing, the axis
order, the attention scale, the value norm and the position tables were all
fixed together, and the embedding then matched to 4e-05 on a peak of 70 —
which is float noise on bfloat16 weights.

**The audio tower needed an off-by-one that no shape could catch.** The
reference masks its chunked attention with `dist < left_window_size`, a strict
inequality, so a query reaches `attention_context_left - 1` keys at or before
itself — twelve, not thirteen. Every tensor had the right shape either way, and
every intermediate looked plausible. A Python replica of the corrected window
reproduced the reference exactly, which is how the bound was settled before the
C was touched.

**The subsampler was right the first time**, matching to 5e-05 on a peak of
112 — the two convolutions, the mean-subtracting norm, the rectifier, the
flatten order and the join projection.

### Reading a difference on a checkpoint that rounds

The residual differences are the same phenomenon 0.4.0 documented for the text
stack, and it is worth showing that they are, because "0.8% off" and "one
quantization step" look identical until they are measured.

On the vision tower's first attention the largest difference was 0.167013
against a calibrated `o_proj` output step of 0.167012 — one step exactly. Of
1.8 million values, 523 differed at all, and **not one** by more than a single
step. On the audio tower's first feed-forward the largest difference was one
step of `ffw_layer_1`'s output, and the 61% of values that then appeared to
differ were the RMSNorm behind it spreading a single flipped step across the
row it normalizes.

So the towers are judged the way the text stack is: against how far the
reference moves when something that should not matter changes. Nudging the
reference's own input by a part in a million moves the vision tower's rows by
2.62 where the engine differs by 2.08, and the audio tower's by 5.36 where the
engine differs by 2.38. In both cases the engine is closer to the reference than
the reference is to itself. On synthetic float weights, which carry no
activation grid, the vision tower agrees to 1.2e-07 and the audio tower to
2.2e-05 — float32 noise.

### The harness

`app_fake.py` no longer builds tower weights by hand. `tower_open` loads one
tower of any checkpoint — shipped or synthetic — from the reference's own
module, and `tower_build` writes a synthetic one using the reference's classes,
so both paths compare against upstream. The torch transcription that 0.5.0 used
as a stand-in is gone; it was a reasonable thing to write when there was nothing
to compare against, and keeping it now would only offer a second opinion that
carries no weight.

`app_diff.py --media` walks either. `--media --model <folder>` is what holds the
engine to the shipped export.

The filterbank is checked separately against the reference's own
`Gemma4AudioFeatureExtractor`, and agrees to 1.4e-03 on a peak of 6.9 with a
mean of 6.6e-05.

### Testing

`test_tower` was rewritten around the reference's names and arrangement, and its
in-test definition of the vision tower with it. The audio tower is no longer
transcribed a second time in C: a conformer written out twice would mostly be a
copy, and the real reference is now available to do better. What it keeps are
checks on the pieces peculiar to it, the end-to-end run, the causal property
that a longer clip repeats a shorter one's first row, and the requirement that a
clip which sounds different reaches a different answer.

Two of those tests failed when first written, both because the test was wrong
rather than the engine: the frame count no longer follows the old formula now
that a frame is cut one sample longer than its window, and a longer clip *must*
repeat a shorter one's first row, because the conformer's attention only reaches
backwards. Both are now asserted in the direction that says something.

The suite is 285 assertions and builds clean under `-Wall -Wextra`.

### Also true now

The text stack runs on the shipped checkpoint and produces sensible prose. That
had been recorded as verified in 0.4.0 but could not be re-run here, for the
memory reason above; what can be said from this host is that the engine loads
the real export — 35 layers, a vocabulary of 262144, 2.3 GiB of weights — and
answers a question about gravity correctly at 5.5 tokens a second.

`token_decode_book` left its output slot untouched when it returned zero for a
control token, so a caller reading the text without checking the count printed
uninitialized stack. Every media prompt begins with a run of placeholder ids,
which made it show up immediately.

### Known gaps

The whole multi-modal graph has not been run against the reference end to end,
only each tower separately, because `Gemma4ForConditionalGeneration` does not
fit in memory here. The seam between them — the placeholder run the processor
lays down, and the ids around it — is the part that remains unchecked.

The engine emits whatever soft tokens a clip yields; the reference's processor
pads or trims to a fixed count, and its feature extractor produces one more
frame than the engine on a clip that does not divide evenly. Nothing in the
towers has been optimized: an image at the full patch budget spends its time in
a triple loop over 2340 patches.

The sanitizers were not run over any of this, for the same reason as before.
`TODO.md` has the rest.

---

## 0.7.0 — the seam

### Why

Two towers agreed with the reference and a text stack agreed with the reference,
and neither result said anything about where they meet. Version 0.6.0 recorded
that gap as the first open item: the placeholder run the processor lays down, and
the ids around it, had never been compared to anything. The reason given was that
`Gemma4ForConditionalGeneration` will not load — the shipped export's language
model dequantizes to about nineteen gigabytes — which is true and turned out not
to matter, for the same reason it did not matter for the towers: when a model is
too large to hold, the piece under suspicion usually is not.

The seam splits into two questions that can be reached separately.

**Where the ids go** is the processor's answer, and a processor is a tokenizer
and three small configurations. It needs no weights at all, so that half runs
against the shipped export as easily as against anything else.

**What the graph makes of them** needs the whole model. A synthetic one fits: the
reference's own model class writes the arrangement, both towers, both projectors
and the names, and `app_fake.whole_build` adds the tokenizer and the processor
files beside them so that `AutoProcessor` and `Gemma4ForConditionalGeneration`
both open the folder with nothing adapted by hand.

### What the comparison found

**The run was laid down bare.** `Gemma4Processor.replace_image_token` returns
`boi_token` + one `image_token` per soft token + `eoi_token`, and
`replace_audio_token` the same with `boa_token` and `eoa_token`. The engine wrote
the placeholders and neither bracket. Those two ids are ordinary text to the
model — it embeds them, attends to them, and they shift every position after
them — so a prompt without them is a prompt the reference never sees. The shipped
config records all four (`boi_token_id` 255999, `eoi_token_id` 258882,
`boa_token_id` 256000, `eoa_token_id` 258883, which one export spells
`eoa_token_index`), and the vocabulary carries them under `<|image>`, `<image|>`,
`<|audio>` and `<audio|>` when a config omits them.

**The per-layer embedding read the placeholder.** 0.6.0's comment said the
reference feeds the placeholder id to the per-layer embedding too. It does not.
`Gemma4Model.forward` builds `llm_input_ids` by rewriting every media position to
`text_config.pad_token_id` and computes the token-identity half of the per-layer
input from that; only the context half sees the tower's row, through the merged
embeddings. So nothing of the placeholder reaches the stack. Removing the fix
again moves the top eight of the synthetic checkpoint's distribution into a
different order, which is how much it was worth.

**The metaspace mark was applied per call rather than per chunk.** The reference
splits its input on the special tokens and normalizes each chunk on its own, so
the mark lands after a special id and nowhere else. The engine applied it at the
start of every `token_encode_book` call, which put one in front of the user's
words when they merely continued the line the role marker opened. On a tokenizer
whose scheme is `always` — the synthetic one — that is one spurious id in every
chat-framed prompt. `lead_marker` now means "this text begins a chunk", which is
what `token_split` needed all along and what the two callers were already trying
to say.

None of the three could have been found one tower at a time.

### The result

`run.py parity --seam` walks four cases — no attachment, one picture, one clip,
and both in the order a content list gives them — and reports, for each, whether
the ids are the reference's id for id, whether the soft-token counts are what the
processor's own arithmetic asks for, and how far the distribution differs.

On the synthetic checkpoint every case agrees: the ids exactly, the picture's 63
soft tokens against the 63 the aspect-preserving resize asks for, and the logits
to between 5e-07 and 6e-05 where nudging the reference's own input by a part in a
million moves it by up to 2e-06. The seam carries no activation grid of its own,
so what is left is float noise.

The clip's count is reported rather than judged. How many rows the engine's own
frames become is the seam's arithmetic and the ids assert it; how many frames the
clip should have made is the feature extractor's framing, which is the next item
in `TODO.md`.

### The harness

`app_fake.whole_build` writes the whole model: text stack, both towers, both
projectors, a tokenizer carrying the turn markers and the six media tokens, a
chat template that is the shipped one's user turn and nothing else, and the
processor files. Two things had to be made honest rather than convenient. The
soft-token budget is one of the five the image processor accepts, so the
synthetic checkpoint uses the smallest, 70, instead of a number of its own. And
the audio window is now written twice, in samples for the engine and in
milliseconds for the reference, because the reference derives its window from a
duration and was quietly using a default while the engine used the stated
sixteen — two extractors described in one file, agreeing about nothing.

The towers are fed the rows the engine says it read, as `--media` feeds them, so
what is measured is the seam and not the png reader or the filterbank, both of
which have independent definitions in `app_test.c` already.

### Testing

`test_tower` gained the bracketing — that a run is an opener, one placeholder a
row, and a closer; that each span is reported past its opener, where the rows go;
that the words after a run are marked as a fresh chunk and the turn then closes —
and one property that only holds now: priming the same rows under a different
placeholder id reaches the same logits to the last bit, because the id under a
filled lane reaches nothing. The tokenizer test that asserted a prepend
normalizer marks the lead word "whatever the caller asks" now asserts the
opposite, which is what the reference does.

The suite is 298 assertions. `run.py parity`, `--media` and `--seam` all pass,
and the engine builds clean under `-Wall -Wextra`.

The sanitizers have now been run, which 0.5.0 and 0.6.0 both had to leave
undone for want of a toolchain that ships them. `run.py test --debug` passes all
298 assertions under the address and behaviour sanitizers, and so does a
multi-modal prompt through the command line — a picture, a clip and a turn frame,
which is every line the media path has.

### Known gaps

The seam has not been run against the shipped checkpoint. It needs no weights to
run the layout half there, and the export could not be fetched on this host.

`run.py install` now asks for `torchvision` and `pillow` as well: the reference's
image processor is a torchvision backend, and a processor cannot be opened
without it. `TODO.md` has the rest.

---

## 0.7.1 — the seam on the shipped checkpoint

### Why

The seam landed with its layout half unrun against the real export, because the
host it was written on could not fetch the weights. This one can: they are
vendored under `model/`. Running it there was the whole of the open item, and it
passes — but getting to that answer needed two faults fixed and one belief
withdrawn.

### It passes

```
text only        ids ok  14
one image        ids ok 276    soft tokens ok  260 = 260
one clip         ids ok  54    soft tokens ok   38 =  38
image and clip   ids ok 316    soft tokens ok  both
```

Every id the engine lays down for a multi-modal prompt on the shipped export is
the id the reference's own processor lays down, and every run of soft tokens is
the length the processor asks for. That is the join the towers and the text
stack meet at, and it had never been compared on real weights.

### The graph half was reporting a false failure

The whole-graph comparison called the shipped checkpoint wrong on a prompt with
nothing attached: a largest logit gap of `2.965e-01` against an allowance of
`1e-3`. The allowance was that small because the floor came back as exactly
zero, and it came back as zero by construction — with nothing attached there is
no input to nudge, and the code reasoned that "the ids are exact on both sides
and there is nothing for a grid to round".

The first half of that is true and the second does not follow. Identical ids do
not make the arithmetic exact on a checkpoint that rounds every activation onto
a static grid: a sum landing on a half step falls one way here and the other way
there, which is the whole subject of 0.4.0. A floor of zero was not a
measurement, it was the absence of one.

The measure that applies is already in this file, used on the text stack: feed
the same tokens again in two chunks behind the cache, which is the same
arithmetic in another summation order. With that in place the verdicts are what
the numbers actually say:

| | engine differs by | the reference moves |
| --- | --- | --- |
| text only | 2.965e-01 | **5.330e+00** |
| one image | 5.692e-01 | **3.271e+00** |
| one clip | 5.370e-01 | **2.696e+00** |

In all three the engine is nearer the reference than the reference is to itself
under a change that should not matter — by a factor of five to eighteen.

A floor that cannot fail is worth no more here than an oracle that cannot fail
was worth in 0.3.0.

### And crashing

The fourth case took the run down with an out-of-memory error partway through,
losing the three results before it. The graph half now catches that and reports
a skip with the reason, so a host that cannot hold two forwards still gets the
ids and the soft token counts — which is most of what the seam is for. The gap
that case measures, `4.051e-01`, is in line with the three that pass; it is
reported and left unjudged rather than compared against a bar nothing measured.

### The model does fit, and never did not

Both the tower work in 0.6.0 and the seam in 0.7.0 were scoped around a claim of
mine: that the shipped export's language model "dequantizes to about nineteen
gigabytes and will not fit on an ordinary machine". That is false, and it was
never true. The figure is the footprint the weights would have if they were
unpacked at load, which is not what happens — `replace_with_quant_layers` leaves
them packed and decodes per forward. Measured:

```
Gemma4ForConditionalGeneration.from_pretrained("model", dtype=float32)
    loaded in 3 s, 2.34 GiB of resident parameters
```

with the process at about 3.1 GB while the whole model is live, and a transient
of roughly a gigabyte and a half per forward, because reading one row of an
embedding dequantizes the whole table. That last part is why two forwards at
once are tight; it is not why nineteen gigabytes were ever needed.

The claim is removed from `app_diff.py`, `app_fake.py`, `GUIDE.md` and
`README.md`. What it was used to justify — building one tower alone rather than
standing the whole model up — is kept, because that is still the better way to
look at one encoder: quicker, lighter, and it keeps a tower's arithmetic
isolated from everything around it. It just is not a necessity.

The entries for 0.6.0 and 0.7.0 are left as they were written. They are the
record of what was believed at the time, and this is the correction.

### Known gaps

The seam's last case is unjudged on the shipped export for want of memory, and
`TODO.md` carries it. Nothing else changed: the suite is 298 assertions, the
synthetic seam still passes all four cases with the new floor, and the towers
are where 0.6.0 left them.

---

## 0.7.2 — the towers get faster and the framing gets right

### Why

Three open items, taken in order of what running the last release had exposed:
the seam's last case went unjudged, the audio framing was known to differ from
the reference by a frame, and an image cost four and a half minutes. The first
is closed, the third is much better, and the middle one turned out to be correct
as it stood — which took a wrong fix and a failing seam to establish.

### An image is two and a half times quicker

Vision attention was the whole cost of a picture and had never been touched. At
the shipped export's patch budget an image is two thousand three hundred and
forty patches scored against themselves, sixteen layers over — about six hundred
billion multiply-adds — and it ran on one core in one loop.

Two changes, neither of which moves a number:

**It goes to the pool.** The query loop is the one place in either tower that
divides cleanly: a band writes only its own rows of the blend and its own slice
of the score scratch, and reads everything else.

**A head is gathered before it is scored.** Where the projections leave them, a
head's keys are sixty-four floats in every seven hundred and sixty-eight, so
scoring one query walks the whole seven megabyte array and the next query walks
it again. Copying one head into a run of its own turns that into six hundred
kilobytes that stays in cache across every query in the band. The copy costs a
few hundred megabytes of memory traffic per image and saves a great deal more.

Together: four minutes thirty-seven to one minute fifty-two on four cores, with
the tower's output identical to the last digit — `2.078e+00` against the
reference before and after.

The first attempt was wrong in a way worth recording. `pool_run` hands out
`worker_count + 1` slices, because the thread that calls it takes one rather
than waiting idle; the scratch was sized at `worker_count` and the last band
wrote off the end of it. It survived a run and a timing, and only fell over
under the trace build. The rule now lives in `pool_bands`, which both `pool_run`
and its caller use, so there is nothing left to get wrong twice.

### The audio framing, and a padding that must not be copied

This one was implemented backwards first, and the seam caught it, so it is
written down the way it happened.

`Gemma4AudioFeatureExtractor` pads a clip out to a multiple of a hundred and
twenty-eight samples before it frames anything — a default of its `__call__`
rather than a field any configuration writes down, which is why it had been
missed. The engine made a hundred and forty-nine frames on a clip where the
extractor made a hundred and fifty. That reads as a plain omission, and it was
fixed as one: round the clip up, hand the extra frames on as zeros.

The synthetic checkpoint's seam case then went from agreeing with the processor
at sixty-three soft tokens to claiming sixty-four.

The padded frames are masked, and masked all the way down. `input_features_mask`
marks them; `Gemma4AudioSubSampleConvProjection` zeros them at the input of each
of its two stride-two stages; the conformer's attention mask excludes them; and
`Gemma4Model` keeps only the rows the tower's own output mask leaves, in one
line — `audio_features[audio_mask_from_encoder]`. Nothing downstream can see
them. What counting them does change is the number of rows the clip is worth,
because the mask is subsampled as `mask[:, ::2]` twice, so a row survives only
if the frame at four times its index is real. Padding therefore cannot alter a
single value the model reads, and can only add soft tokens the reference never
asks for.

So the measurement was redone against the thing that decides it, over
twenty-two clip lengths from a twentieth of a second to ten seconds:

| | agrees with the extractor |
| --- | --- |
| the engine's frame count against the live count the mask marks | 22 of 22 |
| `ceil(live / 4)` against `_get_num_multimodal_tokens` | 22 of 22 |
| the **padded** frame count against the same token budget | 19 of 22 |

The three it parts on are 800, 4000 and 100000 samples — four thousand samples
reach twenty-four frames and pad to twenty-five, and twenty-four frames are six
soft tokens where twenty-five would be seven.

The engine's original framing was right, and the change has been taken back out.
What the release adds is the reason it is right, in `GUIDE.md` and in the tests,
so that the next reader who notices the extractor emitting one more frame than
the engine does not fix it again.

### Every seam case is judged now

The case with both a picture and a clip in it had never once been judged. Three
of the four ran whole and it reported its gap and stopped, because the
reference's second forward — the one that measures how far it moves against
itself — would not fit beside the first.

Two changes, and the second is the one that mattered.

**A floor that can be measured either way.** The nudge moves the input by a
millionth, which means running the towers a second time. Reordering the sums
instead does not, and measures a change that matters just as little, so the
graph half tries the nudge and falls back to the reordering.

**A process per case.** Which case fitted depended on what had run before it,
which is no way to judge anything: `one image` passed alone and skipped after
`--media` had run in the same shell. So `--seam` now spawns itself once per
case. The engine's work is not repeated — the child reads the trace the parent
already wrote — so the cost is one model load, about three seconds, per case.
The parent also lets go of that trace before the child starts: it is half a
gigabyte behind a single picture, it was being held parsed while the child read
the same file again, and three numbers out of it are all the parent needs.

That was worth doing carefully, because the first attempt hid a failure. A child
that dies returns 1, and 1 was also what the parent read as "one check
disagreed", so a case that crashed was counted as a fault and printed nothing at
all — a silent `1 checks failed`. The child now answers 0 or 2 and never 1, and
a child that dies has its last lines printed, sorted into the host running out
of memory and anything else, which is this harness's fault and says so.

### What the seam says on the shipped export

All four cases, every check, on the real weights:

| | engine differs by | the reference moves | |
| --- | --- | --- | --- |
| text only | 2.965e-01 | 5.330e+00 | 14 ids |
| one image | 5.692e-01 | 3.271e+00 | 276 ids, 260 soft tokens |
| one clip | 5.370e-01 | 2.696e+00 | 54 ids, 38 soft tokens |
| image and clip | 4.051e-01 | 2.296e+00 | 316 ids, 260 and 38 soft tokens |

Every id agrees, every soft token count agrees with what the processor asks for,
rank one agrees in all four, the whole top eight is shared in all four, and in
each case the engine sits five to eighteen times nearer the reference than the
reference sits to itself under a change that should not matter.

### Testing

Twelve new assertions pin the frame count at the shipped export's framing — a
320 sample window, a 160 sample hop, half a window of lead — against six clip
lengths the extractor was measured on, including the two that straddle a block
boundary and the one where padding would buy a seventh soft token. The suite is
310 assertions, clean under `-Wall -Wextra` on the SSE2 and AVX2 backends.

### Known gaps

The longest prompt the seam judges is 316 ids; a prompt long enough that the
reference's forward will not fit even alone has not been tried. Nothing in the
conformer has been made faster; its convolution module still walks its kernel
per channel per frame. `TODO.md` has the rest.

---

## 0.7.3 — the parity result, written down

### Why

Everything the suite knew about the engine it knew from weights it had made
itself. The comparison against the real ones lives in `app_diff.py` and
`app_test.py`, and both want python, torch, and a transformers that knows the
architecture. This host is the case that makes the gap plain: it has the
export vendored under `model/`, a compiler, and a transformers with no `gemma4`
in it. Every claim in this file about the shipped checkpoint was, on a tree like
that, unverifiable — not because the checkpoint was missing, but because the
thing that had checked it was.

So what the comparison settled is now recorded in `app_test.c`, where a C
compiler is the only thing needed to read it back.

### What is recorded

Three prompts, run through the whole stack — frame, prefill, one step — and for
each of them the ids of the frame and the head of the distribution:

| prompt | ids | rank one | its lead |
| --- | --- | --- | --- |
| The capital of France is | 14 | 818 | 3.27 |
| Say hello. | 12 | 9259 | 4.36 |
| List the first three prime numbers. | 16 | 818 | 4.31 |

The frame has to come back exactly, rank one has to come back exactly, the
eight highest logits have to come back within 2.0, and none of those eight may
fall out of the top sixteen.

### The slack is a measurement

A fixture that demands the last bit would be a fixture for one compiler on one
host. The checkpoint rounds every activation onto a static grid, which is the
subject of 0.4.0: a sum landing on a half step falls one way in one build and
the other way in another, and the difference is a whole step rather than an
ulp. So the bar had to be measured before it could be set.

Between this host's SSE2 build and its AVX2 one, over five prompts:

| | |
| --- | --- |
| rank one moved | never |
| the top eight logits moved by at most | 1.30 |
| the lowest rank an id in a top eight fell to | 11 of 16 |

The bar is 2.0 and sixteen ranks — above what a backend change did, and inside
the 2.30 to 5.33 the reference moves against itself when its own input is
nudged by a millionth (0.7.2). What that catches is a layer wired wrong, not a
last-bit difference; per-tensor equality is not the criterion here for the same
reason it was not the criterion in 0.4.0.

Held to it, the AVX2 build reproduces the SSE2 recording with room to spare —
the worst of the three prompts moves a logit by 1.107 and drops an id to rank
twelve.

The prompts were picked for the same reason the bar was measured. Of eight
candidates, the three kept lead their runners-up by 3.27, 4.36 and 4.31, all
more than twice the widest move a backend change was seen to make. Two that
read as well were dropped for leads of 1.01 and 1.03, and one measured earlier
led by 0.28. A lead inside the noise is a coin toss dressed as an assertion,
and both backends happening to call it the same way is not a reason to write
it down.

### What it costs, and what is not in it

About twenty seconds at `-O3`, a minute unoptimized, on top of a suite that ran
in a seventh of a second. That is three forwards of a two-and-a-third gigabyte
model and there is no way around it. The section is skipped, with the reason
printed, when there is no checkpoint beside the tree; `IGLLM_MODEL` names one
elsewhere, and naming a folder that holds no `config.json` — `IGLLM_MODEL=none`
— turns the section off. A folder holding some other checkpoint is passed over
rather than failed: the shape is checked against the export's own first, and
all a failure there would report is that these numbers are not about it.

No picture is in it. The processor lifts even a thirty-two pixel image to the
export's full patch budget — 260 soft tokens, four and a half minutes on this
host — so the cheapest media shot would cost more than the whole of the rest of
the suite by three orders of magnitude. That join is judged by
`run.py parity --seam --model model`, which is where it belongs.

### Testing

Twenty-two new assertions, and the suite is 332 when the checkpoint is beside
it and 310 when it is not. Clean under `-Wall -Wextra` on the SSE2 and AVX2
backends, and passing on both from one recording.

### Known gaps

The recording is of the engine, not of the reference: it says the stack still
reaches what it reached when the reference last judged it, not that the
reference would say so today. Re-judging still wants python. `TODO.md` has the
rest.

---

## 0.7.4 — the batch stops losing to the thing it exists to beat

### Why

Throughput had never been looked at. The first thing looking at it found was
not a slow kernel but an inverted one: `bench` reports prefill and decode side
by side, and prefill — thirty-seven tokens run in lanes, which is the whole
point of the batched path — came in at 2.05 tokens a second against decode's
5.65, one token at a time. Batching was costing nearly three times what not
batching cost.

### Where it went

Two kernels, both single-lane against many, timed on this host at the shipped
export's own shapes:

| 12288 x 1536, one thread | one lane at a time | sixteen lanes batched |
| --- | --- | --- |
| SSE2, four bit | 0.075 s | **0.266 s** |
| SSE2, two bit | 0.075 s | **0.266 s** |
| AVX2, four bit | 0.039 s | 0.051 s |
| SSE2, eight bit | 0.006 s | 0.006 s |

`kern_dot_code` decodes packed weights inside its own vector loop: a nibble
unpacks with a shift and a mask, never becoming a float in memory.
`kern_row_code_many` did the opposite — it called `pack_read` once per weight,
scalar, to build a float scratch every lane could then read. The comment above
it said a batch "pays the decode cost of a single vector", and in count that was
true; in cost it was not, because the decode it paid was the slow one. Sixteen
lanes bought one decode and gave back sixteen vector loops.

The four bit AVX2 row shows the same thing in the small: the batch is nearly as
quick as the single lane, which for sixteen times the work means the spread is
all of it.

### `kern_code_spread`

The fix is to write the decode out once, properly: the same shift-and-mask
sequences `kern_dot_code` fuses into its loop, storing floats instead of
accumulating. Two, four and eight bits, on AVX2, SSE2 and NEON, behind the same
macro layer as everything else.

| 12288 x 1536, sixteen lanes | before | after |
| --- | --- | --- |
| SSE2, four bit | 0.266 s | 0.061 s |
| AVX2, four bit | 0.051 s | 0.032 s |

Batching now wins where it should: 1.24 times a single lane on SSE2 rather than
0.28.

### The accumulator was the pace-setter

With the spread fixed, the kernels were still slower than their instruction
counts implied. Every vector path in this file accumulated into one register.
A four bit SSE2 row is four multiply-adds per sixteen codes, chained: each waits
on the last, so the loop ran at the latency of an add rather than at its
throughput. The arithmetic was cheap enough that the dependency was the cost.

Two accumulators, summed at the end — one extra register, no change of method:

| 12288 x 1536, one thread | one chain | two chains |
| --- | --- | --- |
| SSE2, four bit, one lane | 0.075 s | 0.047 s |
| AVX2, four bit, one lane | 0.039 s | 0.027 s |
| SSE2, four bit, sixteen lanes | 0.061 s | 0.039 s |
| AVX2, four bit, sixteen lanes | 0.032 s | 0.018 s |

This is the only change here that touches decode, which runs one lane and so
never went near the spread.

### Eight bits had no vector path at all

Two and four bits had one. Eight did not — it was a scalar byte loop, six times
slower per weight than the four bit vector path beside it. It is the width the
vision tower is quantized to, and the width of the two per-layer projections the
text stack runs every layer. Vectorizing it is the same sequence as the others
with the unpacking removed: load sixteen bytes, apply the flip a vector at a
time, widen, multiply. A 256 x 1536 row went from 0.006 s to 0.001 s.

### What it comes to

Thirty-seven tokens of prompt, sixteen decoded, four cores of a 2017 desktop:

| | prefill before | after | decode before | after |
| --- | --- | --- | --- | --- |
| SSE2 | 2.05 tok/s | **12.34** | 5.65 tok/s | **7.28** |
| AVX2 | 7.37 tok/s | **21.26** | 10.40 tok/s | **13.45** |

Prefill is six times quicker on the default build and prefill now beats decode
per token, which is what a batch is for. Decode is a little under a third
quicker on both. And an image — thirty-two pixels square, which the processor
lifts to the full patch budget of 260 soft tokens, then 316 ids of prefill
behind it — went from 4 m 28 s to 1 m 02 s on the SSE2 build, because the tower
is eight bit and runs its projections in lanes.

### Nothing moved that should not have

Every change here alters the order of a floating point sum, which on this
checkpoint moves a logit by whole steps of the activation grid. That is what
0.7.3 recorded the shipped export's answers for, and the recording was left
where it was rather than taken again from the faster build: re-recording would
launder an unjudged change into the record. Held to the numbers the judged build
produced, the optimized build moves a top-eight logit by at most 1.176 against
an allowance of 2.0, and drops no recorded id below rank nine of sixteen —
which is where an AVX2 build already sat, at 1.107. The suite is 332 assertions
on both backends.

### Known gaps

Decode gained a third and no more, because it runs one lane and its cost is the
fused decode inside `kern_dot_code`, which is now dependency-free but still
spends about two instructions a weight on the narrow widths. A byte-indexed
table of unpacked floats would spend less; it is not written. `TODO.md` has the
rest.

---

## 0.7.5 — the clip stops where the processor stops

### Why

The framing was settled in 0.7.2: the engine's frame count is the live count
`input_features_mask` marks, checked against the extractor on twenty-two clip
lengths, and the rows it makes are `ceil(live / 4)` on all of them. What was not
settled was the ceiling.

A clip is not only framed, it is budgeted, and the budget was in a file the
engine had never opened. `processor_config.json` records an `audio_seq_length`
of 750 and an `audio_ms_per_token` of 40 — 750 soft tokens of forty milliseconds
each, half a minute of audio — and the processor pads or trims a clip to that
before the tower is handed anything.

Nothing had parted the two, because every clip that had been tested was inside
the ceiling. Past it they part, and not by a little: thirty-five seconds frames
to 3499 live frames and 875 rows, where the processor writes 750 placeholders.
The engine would have laid a hundred and twenty-five rows on ids that were never
there.

### Only the trim is visible

The other half of what the processor does — padding a short clip out to the
budget — is invisible here, and it is worth saying why rather than matching it.
The padding is marked in `input_features_mask`, zeroed between the subsampler's
two convolution stages, excluded from the conformer's attention, and dropped
from the rows the tower returns. A short clip is therefore worth its live frames
on both sides, which is exactly what the engine already computed. Matching the
padded shape would change nothing except the arithmetic that decides how many
soft tokens to ask for, and 0.7.2 established that counting padded frames there
is what gets that wrong.

The trim is the half that reaches the model, so the trim is the half that is
implemented.

### Where it cuts

`config_budget_read` reads the pair beside `config_sound_read`, which already
reads the analysis window out of `preprocessor_config.json`; both default to the
shipped export's values, which is the only thing a checkpoint that writes
neither file can be read as meaning. `sound_budget_samples` turns the pair into
a sample count, and `media_audio` applies it to the resampled clip **before** a
frame is taken from it.

Cutting the samples rather than capping the rows is the whole of the design. A
cap would agree with the reference on the count and disagree on the last row,
whose frames would have been drawn from audio the reference stopped reading. The
suite states it that way round: a clip four times the budget has to come out
equal to the clip that ends at the budget, row for row, not merely the same
length.

The two halves of the processor's configuration agree that this is the right
place to cut. Seven hundred and fifty tokens of forty milliseconds is 480,000
samples at sixteen kilohertz; those frame to 2999 live frames at a 320 sample
window and a 160 sample hop; and `ceil(2999 / 4)` is 750 — the budget exactly,
and not a row over. The framing and the budget are written by different parts of
the export and they meet on the same number.

### Saying so

A clip that is cut is a clip whose tail the answer will not be about, so
`app_media` carries a `cut_flag` and the command line prints one line when it is
set. `model_audio_rows` and `model_audio_span_ms` expose the ceiling beside
`model_image_rows`, and `probe` reports it:

```
audio    yes
  rows   750 at 40 ms, 30.0 s of clip, placeholder id 258881
```

### What it comes to

A thirty-five second clip through the shipped export, `tokens` task:

| | soft tokens | prompt ids |
| --- | --- | --- |
| before | 875 | 883 |
| after | **750** | 758 |

and the run says `audio: the clip runs past the budget of 30 s and is cut to it`.

### Testing

The fixture grew a third configuration file. Four frames of four samples at
eight kilohertz is one soft token there, which makes a token two milliseconds,
and a budget of sixty-four of them is one a fixture can be written past without
the tower costing anything: a clip at the budget and a clip four times as long
are compared row for row, and the shorter of the two is checked to have spent
the whole budget without being cut. The shipped export's own ceiling — 750 at
40 — is asserted where the recorded prompts are.

Nine new assertions. The suite is 341 when the checkpoint is beside it and 318
when it is not, clean under `-Wall -Wextra` and passing on the SSE2 and AVX2
backends.

### Known gaps

The reference has not re-judged this. `run.py parity --seam` wants a
`transformers` carrying `Gemma4Processor`, which is not installed on the host
this was written on, so what is checked here is the export's own recorded budget
and the engine's arithmetic against it — not a fresh answer from
`_get_num_multimodal_tokens` on a clip past the ceiling. That is the check to
run when the reference is next to hand, and it is a cheap one: it is the layout
half of the seam, which needs no weights.

Nothing else about audio moved. Every clip inside the budget frames and counts
exactly as it did in 0.7.4, so the recorded distributions are untouched.
`TODO.md` has the rest.

---

## 0.7.6 — the two bit decode becomes a table read

### Why

0.7.4 left decode a third quicker and said where the rest of it was: the
engine runs one lane in the token loop, so it cannot share a decode with
anybody, and the decode fused inside `kern_dot_code` still spent about two
instructions a weight on the narrow widths. The note in `TODO.md` proposed the
fix — a byte-indexed table of unpacked floats — and this is it.

### What the measurement said first

The proposal is only worth taking if two bits really is the expensive width, so
it was timed before it was written, on the export's widest row and one thread:

| 12288 x 1536, one thread | two bit | four bit | eight bit |
| --- | --- | --- | --- |
| SSE2 | **0.0039 s** | 0.0027 s | 0.0025 s |
| AVX2 | 0.0016 s | 0.0016 s | 0.0014 s |

Two bits is the slowest width on SSE2 while reading a *quarter* of the bytes
eight bits reads. That is an unpacking cost and nothing else, and it is the
width the export leans on hardest.

### The table

`kern_code_two` is 256 rows of four floats — four kilobytes, first level cache
resident — built by a nest of preprocessor macros so it lands in read-only
memory and no kernel has to remember to fill it. Two bits a code means a byte is
exactly four codes, so the unpacking becomes one sixteen byte read.

The values are the codes themselves, in the lanes the shift-and-mask sequence
put them in, and the accumulator each lane is multiplied into is unchanged. So
this is not a close approximation of the old path, it is the same sum in the
same order.

That is the sort of claim worth checking rather than reasoning about, because
every other change to these kernels has moved a logit by whole steps of the
checkpoint's activation grid and needed an allowance written for it. This one
does not: `logits` on the shipped export, whose json is the whole head of the
distribution to six decimal places, is **byte for byte identical** between the
build before this change and the build after it, on both the SSE2 and the AVX2
backend. There is no allowance to spend and nothing to re-judge.

| 12288 x 1536, one thread | unpacking | table |
| --- | --- | --- |
| SSE2, fused dot | 0.0039 s | **0.0023 s** |
| SSE2, spread | 0.0031 s | **0.0023 s** |
| plain loop, spread | 0.0092 s | **0.0023 s** |
| AVX2, fused dot | 0.0016 s | 0.0018 s |
| AVX2, spread | 0.0015 s | 0.0023 s |

### Where it is not used, and why

The wide path keeps its shifts. Filling one 256 bit vector from the table costs
two narrow loads and an insert, against one broadcast, one variable shift and
one mask; it measures slower, so it is not taken. Writing it down rather than
quietly leaving the AVX2 path alone is the point of the row above — the table
looks like it should win everywhere and it does not.

The fused dot's remainder loop keeps its shifts too, for a different reason:
there the cost is a chain of four dependent adds per byte rather than the
decode, and swapping in the table moves it by nothing (0.0179 s against
0.0178 s). The spread's remainder does take the table, where the same swap is
four times quicker, and that is the loop a host with no vector path at all
runs for the whole width.

NEON is left as it was. Its unpacking is already a table instruction — `vtbl1_u8`
across a register — and there is no ARM host here to measure a change on.

### What it comes to

Thirty-two tokens of prompt, a hundred and twenty-eight decoded, four cores of a
2017 desktop:

| | prefill before | after | decode before | after |
| --- | --- | --- | --- | --- |
| SSE2 | 11.86 tok/s | 12.43 | 7.06 tok/s | **9.48** |
| AVX2 | 20.56 tok/s | 21.19 | 13.07 tok/s | 13.24 |

Decode is a third quicker on the default build and unmoved on the tuned one,
which is what the kernel timings predict. Prefill barely moves on either, and
that is also what they predict: a batch pays the spread once and then runs
sixteen lanes over what it left, so a third off one sixteenth of the work is
not visible.

### Testing

The table is checked against `pack_read`, byte by byte and code by code, which
is the suite's usual rule — an independent account of the same thing rather than
a recording of this one. It earns its own assertion because a nest of macros
that builds 256 rows is either right or catastrophically wrong, and every kernel
that reads it would agree with itself either way.

One new assertion. The suite is 342 with the checkpoint beside it and 319
without, clean under `-Wall -Wextra` and passing on the SSE2 and AVX2 backends.

### Known gaps

Three, five, six and seven bits still walk the bit stream in both functions, and
the same table trick does not reach them — those widths do not divide a byte, so
a byte is not a whole number of codes and the index is not a byte. Four bits
does divide, but a byte is two codes there and the table would be read twice as
often for half as much; it was not tried, because four bits already measures
within a tenth of eight bits, which is the floor.

And the floor may be the memory rather than the arithmetic. This export carries
no mixture-of-experts block, so decode reads close to the whole two and a third
gigabytes of weights for every token it produces; 13.24 tokens a second is about
thirty-two gigabytes a second, which is near what a desktop of this age will
hand over. If that is where the tuned build now sits, the next gain is in
reading fewer bytes rather than in spending fewer instructions on them. Nothing
here has measured that either way. `TODO.md` has the rest.

---

## 0.8.0 — attachments where the words are, and the wall decode is against

### Why

Three of the open tasks could only be settled where the shipped export is, and
this is a pass over those. Two of them were waiting on a reference that would
run at all: the `transformers` on the host 0.7.5 and 0.7.6 were written on
carried no `gemma4`, so the clip budget and the ids around a run were held to
the export's own configuration rather than to the reference's answer about it.
A `transformers` that carries one is installed now, in an environment of its own
so that nothing else on the host had to move to make room for it, and it
answers.

### The clip budget, answered by the reference

0.7.5 cut a clip at the processor's budget — 750 soft tokens of forty
milliseconds, cut out of the samples rather than off the rows — and said in its
own known gaps that the reference had not re-judged it.
`_get_num_multimodal_tokens` is the processor's answer to the same question, the
one a serving stack asks before it allocates, and it agrees on every length put
to it:

| clip | the processor asks for | the engine makes |
| --- | --- | --- |
| 5 s | 125 | 125 |
| 30 s, which is the budget exactly | 750 | 750 |
| 35 s | 750 | 750 |
| 61 s | 750 | 750 |

A picture is the same story on the other tower: a 320 by 240 png is 266 soft
tokens to the processor's aspect-preserving resize and 266 rows to the engine.

Nothing was changed to make that true. It is the arithmetic 0.7.5 wrote, put
against the code it was written from, and the gap is closed rather than fixed.

### More than one picture or clip

The engine never cared how many runs a prompt carried. `token_media_run` walks a
list of spans, and the substitution takes any set of positions. The command line
cared: `--image` and `--audio` held one path each, and the picture always led
because there could only be one of each.

Both flags are repeatable now, up to eight attachments in a prompt, and the
order they are laid down in is the order they were given in — which is what a
content list means by order, and what a reversed pair has to mean if the flags
are to mean anything at all. So `--audio clip.wav --image photo.png` puts the
clip first, where before it would have put the picture there.

What settles it is the reference's own layout rather than a reading of it. The
seam comparison walks seven cases now instead of four, and on the shipped export
every id of every one of them is the reference's:

| case | ids | runs of soft tokens |
| --- | --- | --- |
| one image | 282 | 266 |
| one clip | 141 | 125 |
| image and clip | 409 | 266, 125 |
| clip and image | 409 | 125, 266 |
| two images | 537 | 266, 253 |
| two images, a clip | 664 | 266, 253, 125 |

The two pictures are deliberately different shapes, 320 by 240 and 224 by 448,
so the second run is 253 rows where the first is 266. A harness that measured
one run and used its length twice would agree with itself on two copies of one
picture and be wrong here, and so would an engine that laid the first tower's
rows down on both runs.

The count of a placeholder id is no longer the length of a run, so the harness
walks the ids, cuts them into runs, and asks the processor about each attachment
separately. A prompt that put the right number of rows in the wrong run would
still have the right total.

The graph half of the two new cases is not run, and says so rather than being
run against the wrong input: the activation dump holds one set of rows per tower
and cannot say which attachment they came from. The reversed pair is not a new
case for it, and there it does run — on the synthetic checkpoint a clip before a
picture reaches the reference's distribution, inside the reference's own
movement, which is the end to end form of the ordering claim.

### Where the words go

Putting every attachment in front of the words is one arrangement of a content
list and not the only one, and the reference's template has always laid down
whatever order it was given. `token_frame_media` could not: it took a block of
spans and a block of text. `token_frame_parts` takes the list itself — an
`app_part` is a stretch of words or one of the spans — and walks it in order, so
a picture can sit in the middle of a sentence. The older call is the newer one
with every span first and the words last, and both go through the same body, so
there is one arrangement of the frame rather than two that can drift apart. The
suite writes a prompt both ways and holds the ids to each other.

The one thing that body has to carry across the pieces is whether the next one
begins a chunk. A media run ends in a closing special id and a stretch of words
does not, so the words after a picture take the tokenizer's lead mark and the
first words of a turn — which follow only the role marker — do not. It is a
one-token difference, in a place no test of the runs themselves would look.

On the command line that is `--text`, which takes its place in the order the
flags were typed. `--prompt` keeps its older meaning, the words after everything
else wherever on the line it is written, so every invocation that predates this
lays down exactly what it did before:

| turn | ids |
| --- | --- |
| words, picture, words | 284 |
| words, picture | 280 |
| picture, words, picture | 533 |
| words, clip, words, picture | 409 |

Every id of those is the reference's too, on the shipped export, against the
same template and the same processor as the rest.

### What decode spends its time on

> This subsection is wrong and 0.8.1 replaces it. The tokens a second and
> the sweep below are right; the byte count they are divided by is the key
> and value cache, not the weights. It is left as written because the
> mistake is instructive and because the numbers around it are still good.

0.7.6 named two candidates and measured neither: converting the per-group gains
once instead of per group, or nothing at all — this export carrying no
mixture-of-experts block, so that decode reads close to the whole two and a
third gigabytes of weights for every token it produces.

It is the second. Two measurements say so, both on four cores of the 2017
desktop every other number here was taken on.

The first is what the host's memory will hand over: a read only sweep of a 2.4
gigabyte buffer, three passes, the best of them kept. It has nothing to do with
the engine; it is the shape the engine reads in.

| threads | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- |
| sequential read | 19.73 GB/s | 24.77 | 27.08 | **27.96** |

The second is decode against it. The engine reports 2067.9 MiB resident for this
export and reads all of it for every token, so a token a second is a little over
two gigabytes a second:

| threads | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- |
| decode, tuned | 4.45 tok/s | 8.33 | 11.36 | **13.49** |
| which is | 8.98 GB/s | 16.82 | 22.93 | **27.24** |
| of what the memory gives | 46% | 68% | 85% | **97%** |

At one thread the engine is nowhere near the memory's limit and the arithmetic
is the cost. At four it is against the wall: 27.24 against 27.96 is inside the
spread of the sweep itself.

The default build is the control. It runs the same reads through a narrower
kernel, and it is not at the wall:

| default, SSE2 | 1 thread | 4 threads |
| --- | --- | --- |
| decode | 3.06 tok/s | 9.96 |
| which is | 6.18 GB/s | 20.11 |
| of the memory | 31% | **72%** |

That is the whole shape of it. On SSE2 there is a quarter of the machine still
to win by spending fewer instructions, which is why the two bit table read in
0.7.6 moved that build by a third and moved the tuned one by nothing. On AVX2
there is nothing left to win that way, and the per-group gain mirror that
`TODO.md` offered as the other candidate is worse than nothing there: it trades
memory for instructions on a build whose whole cost is already memory. It stays
in `TODO.md` scoped to the default build, where the trade is still open and the
bytes it adds are what would have to be weighed against the instructions it
saves.

What replaces it as the next thing to try is reading fewer bytes rather than
converting them faster. The loader chooses between a packed plane and a real one
by size, and every plane it unpacks is bytes added to a read that is now the
whole cost. What residency per tensor is worth is the difference between the
2067.9 MiB this export sits at and what it would sit at packed, and that is one
measurement this release did not take.

The gigabytes here are powers of two, as the engine's own report is.

### Testing

The suite grew two cases for what the front end now depends on. The first is
ordering: four spans in a caller's order rather than a picture and then a clip,
one of them empty, held to the ids each run is bracketed with and to where each
run says its rows go. The second is the parts list: the same turn written both
ways, once through the older call and once as a list with the span first, held
to each other id for id, and then the same span with words on both sides of it,
where the words in front have to stand exactly where the opener stood and the
run has to begin that much further in. Neither needs a checkpoint, because none
of it is the checkpoint's.

Nine new assertions. The suite is **351** with the export beside it and **328**
without, clean under `-Wall -Wextra` on the SSE2 and AVX2 backends.

Three comparisons against the reference were run, all of them clean:

| run | what it covered |
| --- | --- |
| `parity --seam` | nine cases on a synthetic checkpoint: every id, every count, and the distribution for the six with at most one of a kind |
| `parity --seam --model model`, two pictures and a clip | the same nine on the shipped export: every id and every count, and the distribution for the four whose forward, and its floor, both fit |
| the processor's own arithmetic | four clip lengths and a picture, against `_get_num_multimodal_tokens` |

### Known gaps

On the shipped export two of the nine get no distribution for want of memory,
and they fail differently: the case with a picture and then a clip cannot
allocate the forward at all, and the case with the clip first allocates it and
then has no room for the floor beside it — the reference moving its own input by
a millionth and running again. The second reports the gap it measured and leaves
it unjudged. Sixteen gigabytes is the constraint rather than the engine, and it
is the same one 0.7.1 wrote down.

A case with two of a kind is judged on its ids alone, here and on synthetic
weights both. The activation dump holds one set of rows per tower and cannot say
which attachment they came from, so the reference would be fed the wrong input
rather than a hard one, and the graph half says so and stands down.

The gain mirror and the residency measurement are both still open, in the
narrower forms above. `TODO.md` has the rest.

---

## 0.8.1 — what a token actually reads

### Scope

The measurement 0.8.0 left open, taken. It says 0.8.0's conclusion was wrong:
decode is not against the memory wall, and it is not close to it.

Nothing about the arithmetic in that release is affected. The tokens a second
are what they were, the sweep is what it was, and every parity result stands.
What was wrong was the byte count the seconds were divided by.

### The number that was not the weights

`memory 2067.9 MiB` in the bench report is `app_total_bytes`, a running total of
what the engine's allocator has handed out since the process began. The weights
are not in it and never were: they are mapped, not allocated, and `mem_alloc` is
the only thing that counter counts. What is in it is the key and value cache,
which is sized from the window and not from the prompt:

| `--window` | 4096 | 32768 | 131072 |
| --- | --- | --- | --- |
| `memory` | 330.9 MiB | 723.1 MiB | **2067.9 MiB** |

The run that reported 2067.9 asked for the default window of 131072 and then
filled seven slots of it. So the figure 0.8.0 divided the seconds by was the
cache the run did not use, and the resemblance to the size of the checkpoint —
2334.8 MiB of tensors, close enough to pass — was a coincidence.

The report now says which is which, and the model layer answers the question
directly rather than leaving it to be inferred from an allocator counter:

```
reads   762.6 MiB a token, 10.55 GiB/s
weights 2334.8 MiB mapped
memory  2067.9 MiB allocated
```

### What a step reads

`model_decode_bytes` walks the planes the token loop reads and counts what a
product against each of them streams: the payload, and for a quantized plane the
gains and the zero points beside it. `plane_bytes` and `plane_row_bytes` under
it are the same sum for a whole plane and for one row of it.

Three things separate that from what the export weighs, and all three are large:

- **The embedding tables are indexed, not swept.** This export's per-layer
  embedding table is 262144 by 8960 at four bits — 1120 MiB, 48% of the file —
  and a step reads one row of it. The token table is another 96 MiB read a row
  at a time, and it is not the output head here: this export ships `lm_head`
  untied, so the head is a separate 96 MiB that *is* swept.
- **The towers are not in the token loop.** Vision is 180.3 MiB of the
  checkpoint and audio 143.6, and both are read when a picture or a clip is put
  in front of a prompt rather than once a token.
- **A sharing layer binds no keys or values.** Twenty of the thirty-five layers
  read another layer's cache, so of the 15.8 MiB of `k_proj` and `v_proj` the
  checkpoint ships across the stack a token reads 6.8.

What is left is 759.4 MiB — 796,326,800 bytes — against 2334.8 MiB mapped:

| what | MiB a token |
| --- | --- |
| mlp gate, up, down | 475.3 |
| output head | 97.0 |
| attention q | 63.3 |
| attention o | 63.2 |
| per-layer input gate and projection | 26.5 |
| per-layer model projection | 26.2 |
| attention k and v | 6.8 |
| norms | 1.1 |
| embedding rows | 0.005 |
| **a decode step** | **759.4** |

The engine adds the cache to that as the prompt grows, so the figure the bench
reports climbs a little over a run: 762.6 MiB a token averaged over 64 tokens,
against the 759.4 a step costs at the start of one.

### Decode against the memory, again

Same host, same sweep, same build flags, measured this session rather than
0.8.0's:

| threads | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- |
| sequential read | 16.17 GiB/s | 24.29 | 24.40 | **24.87** |

That is a little under what 0.8.0 measured — 19.73 to 27.96 — on a host with
more running on it, and the difference does not matter to what follows. Decode
against it, tuned, over 64 tokens:

| threads | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- |
| decode | 4.49 tok/s | 8.41 | 11.80 | **14.17** |
| which is | 3.35 GiB/s | 6.26 | 8.79 | **10.55** |
| of what the memory gives | 21% | 26% | 36% | **42%** |

And the default build, which 0.8.0 used as its control:

| threads | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- |
| decode | 3.07 tok/s | 5.82 | 8.26 | **10.17** |
| which is | 2.29 GiB/s | 4.34 | 6.15 | **7.57** |
| of what the memory gives | 14% | 18% | 25% | **30%** |

Against 0.8.0's own ceiling of 27.96 the tuned build is at 38% rather than 42%,
which is the direction that makes the point more strongly, not less.

The scaling says the same thing without any byte count at all, and it was
sitting in 0.8.0's own tables. From one thread to four the memory hands over
1.54 times as much; decode produces 3.16 times as many tokens on the tuned build
and 3.31 on the default. A run against the memory wall cannot scale more than
twice as well as the memory does. That check needs no measurement of what is
read, and it is the one that should have caught this.

### What it changes

0.8.0 concluded that on AVX2 there was nothing left to win by spending fewer
instructions, and scoped two open items around that. All of it is withdrawn:

- **The per-group gain mirror is open on both builds again.** 0.8.0 called it
  worse than nothing on the tuned build, on the grounds that the whole cost
  there was already memory. It is not, so the trade is what it always was —
  memory against instructions — on a build with 58% of the machine unused.
- **Residency per tensor is a footprint question, not a decode one.** What a
  packed residency saves is in the two thirds of the mapped total the token loop
  never touches, and that is worth having on a small host. What it does for
  decode is smaller than it looked, because the planes it would pack are not the
  planes a token reads.
- **There is one unquantized plane in the token loop, and it is meant to be.**
  The per-layer model projection is bf16 at 26.2 MiB, 3.4% of what a token reads
  and the only real plane of any size the decode path touches. The export names
  it in `modules_to_not_convert` beside the patch embedder and the two
  projectors, and ships no packed form of it, so the loader is reading the only
  thing there is to read. That is worth having written down: it looked like a
  gap in the loader and it is a decision of the export's.

The gigabytes here are powers of two, as the engine's own report is, and the
report now says GiB rather than GB so that it says so itself.

### Testing

Sixteen assertions that need no checkpoint. `plane_bytes` and `plane_row_bytes`
are held to what the plane fixture allocated — codes, gains, and zero points,
counted from the test's own arithmetic rather than from the function under
test — over the three bit widths the fixture covers, plus a real plane and an
unbound one.

The mixture case gets an exact identity rather than a bound. A step reads the
`expert_top` experts the router picks, so raising `expert_top` to the whole bank
has to move the figure by precisely the experts that were being left out, and
the test computes that difference itself from the expert planes. Counting the
bank whole and counting none of it both fail it.

Four more with the export beside it: the mapped total and the per-token read,
both recorded exactly, and two bounds that survive a re-recording — a token
reads less than the per-layer embedding table alone, and under a third of what
the export holds. A build that swept either table would fail the bounds long
before the exact totals needed touching.

The suite is **371** with the export beside it and **344** without, clean under
`-Wall -Wextra` on the SSE2 and AVX2 backends.

### Known gaps

The cache figure counts the distinct bytes of a layer's span. The heads of one
group re-read the same key head, so a run of `group_share` heads asks for those
bytes more than once; a group's span is tens of kilobytes at the prompt lengths
here and stays in cache, but at a long context that assumption is worth
revisiting rather than trusting.

The sanitizer build could not be run this session: the MinGW gcc on this host
ships no `libasan` or `libubsan`, so `run.py test --debug` fails at the link.
That is the host's gap rather than the suite's, and it is unrelated to
everything above.

---

## 0.8.2 — the cache scales the export ships

### Scope

The seventy calibration scalars under `self_attn.k_cache_scale` and
`self_attn.v_cache_scale` are read for the first time. They were shipped by the
export, ignored by the reference, and ignored here since 0.4.0. This release
says what they hold, what grid they describe, and what a cache stored on that
grid costs — the last of which is measured on the shipped export rather than
argued.

### What they are

Seventy rank-zero `f32` tensors, all of them in the first shard, one pair a
layer. They are per tensor and static: one magnitude for every channel and
every position of a layer's key or value cache.

Only thirty are bound. Twenty of the thirty-five layers read another layer's
keys and values and allocate no cache of their own, so there is nothing for
their scales to describe. The forty that go unread are not junk, and this is
worth stating because it would otherwise look like a gap: each is bit-exact with
the scale of the layer it reads from — layer 13 for the sliding layers, layer 14
for the whole-attention ones, which is the layer `source_slot` resolves to.
Skipping them is safe, and reading them would be reading the same number twice.

### The grid is eight bit float, not eight bit integer

The scales are a magnitude divided by the largest the grid reaches, so the
denominator has to be known before a scale means anything. A byte's 127 is the
obvious guess and it is wrong. Two things say so.

Layer four's value scale is `0.2857142984867096`, which is the float32 nearest
128/448 **to the bit** — a calibration clamped at a round magnitude, over the
largest normal e4m3 carries. Under 127 it would be 36.2857, which is round in
nothing.

And under 127 the ranges are too small for what a plain prompt already puts in
the cache. Eighteen ids of an ordinary question fill layer 13's value range to
229% of a byte's reach, and six other layers overflow it too. Under 448 the same
run fills it to 65% and nothing clips:

```
layer       k range     k peak   fill        v range     v peak   fill
    0      2.686925     0.8841  32.9%      21.165359     8.3354  39.4%
    4      8.288697     0.3142   3.8%     128.000000     7.1995   5.6%
   13      2.667951     0.8484  31.8%      21.165359    13.7411  64.9%
   14      7.821923     0.3747   4.8%     128.000000     7.5050   5.9%
```

The whole-attention layers sit an order of magnitude wider than the sliding ones
and use almost none of it, which is the export's calibration being cautious
where the cache is longest rather than anything the engine can improve on.

### What it costs

`--cache 8` makes the round trip through the grid in float, at the one point a
row enters the cache. Nothing is stored smaller yet: the cost of quantizing is
paid where it can be measured against the same run without it, which is what a
backend that stored bytes would inherit.

On the shipped export, over five prompts, the next token is never in doubt — the
top id agrees every time and the top probability moves by under 1.5 points. What
does not survive is token-exact reproduction: under greedy decoding four of the
five diverge, at 78 to 102 characters in, at the first genuinely close call. All
four stay correct and on topic afterwards; they say the same thing in different
words.

That is the shape of the trade rather than a verdict on it. The reason to want
it is the footprint, and at the default window it is not small: 1803.0 MiB of
cache becomes 450.8 MiB, a saving larger than the whole checkpoint's mapped
weights.

### Surfaces

A `cache` task prints the table above for a prompt of your choosing, the peaks
beside the ranges they have to fit in, and what the cache costs at full span
both ways. The peaks are tracked whether or not quantizing is on, because what
the range has to cover is a question only a real prompt answers.

`model_cache_scale`, `session_cache_peak` and `session_cache_room` expose the
three numbers to a caller.

### Tests

Nine more, on the grid itself rather than the export: every value it carries
survives the round trip, in the normal range and the subnormal one; a magnitude
over 448 saturates rather than running off into an infinity e4m3 has no room
for; the midpoint between two neighbours rounds to the even significand, as the
hardware conversion does. The rounder was also checked against a grid built
from scratch over 200000 values, and agrees on every one.

The suite is **380** with the export beside it, clean under `-Wall -Wextra` and
under the sanitizers.

### Known gaps

Nothing stores the cache smaller yet. The scales are read, the grid is
implemented and the error is measured, but `key_store` and `value_store` are
still float arrays — a backend that holds bytes is the change that collects the
1352 MiB, and it is a backend change rather than this one.

The divergence measurement is five prompts on one export, text only, at short
context. What quantizing costs when the cache is long enough for the sliding
window to be turning over is the case that matters most for a small host, and it
is untested here.

---

## 0.8.3 — the cache stops being floats

### Scope

0.8.2 read the export's seventy cache scales, implemented the eight bit float
grid they describe, and measured what a round trip through it costs. It stored
nothing smaller: `key_store` and `value_store` were float arrays, and the
release said so under its own known gaps. This one holds them as bytes and
collects the 1352 MiB, and then asks the question 0.8.2 left untested — what
quantizing costs when the cache is long enough for the sliding window to have
turned over.

### A change of storage, not of arithmetic

`cache_code8` names the byte a value lands on and `cache_real8` reads it back:
one sign bit, four exponent bits biased by seven, three of significand, with a
zero exponent field standing for the subnormal range. `cache_pack8` has already
rounded and saturated by the time the encoder runs, so the encoder only names
the byte, and 0x7F — where the format keeps its NaN — is never reached, because
448 saturates one code short of it.

Which storage a layer uses is settled once, in `session_open`, a side of a layer
at a time, and is carried by a 256 entry table filled by `cache_grid_fill`: null
where the layer holds floats, and otherwise the grid times that layer's scale. A
null table is the flag, and it also decides what one stored value costs. A layer
the export ships no scale for keeps floats, so a checkpoint that calibrates
nothing costs nothing for asking, and a byte of zero decodes to a float of zero,
which is what lets one `memset` clear either store.

Folding the scale into the table is the point. An entry is exactly
`cache_pack8(value / scale) * scale`, which is the float the 0.8.2 round trip
stored, so a layer holding bytes reaches the same score as one holding floats —
bit for bit, not to a tolerance. `cache_dot` mirrors the packed dot kernel's
blocking, its pair of accumulators and its horizontal sum term for term to keep
it that way: on AVX2 the table read is a gather, on SSE2 and NEON four scalar
reads of a kilobyte that stays in the first level cache.

A layer holding floats never reaches those kernels. It keeps `kern_dot_real` and
the blend loop it always had, so `--cache 8` off is the arithmetic it was before
any of this existed. That is worth stating as a decision rather than an
accident: the first draft of this change routed both storages through one pair
of helpers, and that alone moved the default build's logits enough to change the
top token — the same size of shift the SSE2 and AVX2 builds already differ by,
and no more, but a refactor should not be spending it.

### What it collects

On the shipped export, at the default window:

| | float | bytes |
| --- | --- | --- |
| cache at full span | 1803.0 MiB | 450.8 MiB |
| the process allocates | 2067.9 MiB | 715.7 MiB |

The saving is larger than the checkpoint's own mapped weights. What a decode
step reads falls with it, and `session_cache_bytes` asks the layer rather than
assuming a float, so `bench` reports the fall rather than having to be told
about it.

### Surfaces

`session_cache_room_at` says what the cache costs at either storage;
`session_cache_room` is that at the storage the session actually uses. The
`cache` task prints both, so the trade is on the same line as the ranges.

### Tests

Nine more, on the byte rather than on the grid: every code round trips except
the two the format spends on a NaN, which saturate to 0x7E; the encoder and the
rounding agree over a sweep of forty thousand values and twenty-three powers;
and the table hands back exactly what the float round trip stored, at four
scales, two of them the shipped export's own. The suite is **389** with the
export beside it, clean under `-Wall -Wextra` and under the sanitizers.

The stronger check is not in the suite, because it needs two binaries: against
one built from 0.8.2's tree, the logits and the greedy continuation agree byte
for byte, with `--cache 8` and without it, on the SSE2 and the AVX2 build alike.
That is the claim the folded table exists to make, and it is the reason this
release needs no re-run against the reference: on the default path nothing
moved, and on the quantized path it moved to the same place 0.8.2 measured.

### What a quantized cache costs at length

0.8.2 measured the divergence on five short prompts and said plainly that the
case which decides whether the footprint is worth taking — a context long enough
for the sliding window to be turning over — was untested. It is tested now.

Two prompts of ordinary English prose, 694 and 588 ids, both past the 512 the
sliding layers hold, each decoded greedily for 96 tokens with the cache held as
floats and again as bytes:

| | 0.8.2, five prompts at 7 to 18 ids | here, 694 ids | here, 588 ids |
| --- | --- | --- | --- |
| greedy agrees for | 78 to 102 characters | **17** | **23** |

That is the answer, and it is worse than the short case by a factor of four to
five. Both continuations stay fluent and on topic afterwards and say different
things: one carries on about `app_core.c` where the other turns to `app_main.c`.

What has not decayed with them is the head of the distribution, and the two want
saying together or the first number reads as worse than it is. Taken on the same
702 ids, the next token is not in doubt and is not even close: both storages put
id 2094 first, 28.8084 against 28.6828, over a runner-up 5.16 and 5.22 behind.
The top eight are in the same order on both, and it is rank nine and below where
they begin to trade places, eleven of the top sixteen still in step.

So the divergence at seventeen characters is not the head of the distribution
coming apart at length. It is that a step blending several hundred rounded rows
instead of a dozen needs only one genuinely close call to part the two runs, and
over ninety-six steps it finds one early. Which is the same mechanism 0.8.2
described — it simply arrives four to five times sooner.

What does not happen is clipping, and it is worth recording that it does not.
The peaks a long prompt reaches are higher than a short one's but still well
inside the calibration — the fullest is layer 8's values at 52.2% of its range,
against 26.6% on a seven id prompt, and no layer of the thirty passes 53%. The
export's static ranges are not the thing that gives way at length. The damage is
the rounding itself, accumulated over a span twenty times longer.

So the case for `--cache 8` is footprint, and at length the price is higher than
0.8.2 could see: 1352 MiB saved against a greedy continuation that parts company
four to five times sooner than the short prompts suggested. On a host that has
the memory it is not worth taking. On one that does not it is the difference
between running and not, and the first token is unaffected either way.

### What it costs to read

What a decode step reads is not timed, it is counted from the shapes, so it is
exact and it is the same on every host. Over the 96 decode steps behind the 694
id prompt:

| | float | bytes |
| --- | --- | --- |
| a decode step reads | 807.7 MiB | 771.5 MiB |
| of which the cache is | 48.3 MiB | 12.1 MiB |

A step reads 759.4 MiB of weights whichever way the cache is held, so the cache
is the rest — and 48.3 against 12.1 is a quarter to the tenth of a mebibyte,
which is the arithmetic working. The 588 id case is 804.8 against 770.8, the
same quarter of a slightly shorter span.

What that buys in tokens a second is another matter, and the host will not give
a number. This was written on a shared four core virtual machine, and over the
session the same build in the same configuration — the SSE2 default with a float
cache, on the same prompt — decoded at 4.52 tok/s in one window and 2.32 in
another. A control that moves two to one cannot measure an effect smaller than
that, so no rate is quoted here.

What the runs do agree on is a sign, and only on one of the two builds. Five
pairs were taken, each pair adjacent in its own window:

| build | float | bytes |
| --- | --- | --- |
| tuned, AVX2 | 3.15 | 2.16 |
| tuned, AVX2 | 5.53 | 3.76 |
| tuned, AVX2 | 5.40 | 2.25 |
| default, SSE2 | 4.52 | 4.18 |
| default, SSE2 | 2.32 | 4.09 |

All three tuned pairs put the byte cache behind, by a third to a half. The two
default pairs disagree with each other about which way it goes at all. Three of
one sign is not a measurement and the spread across them is far too wide to
average, but it is the direction the trade predicts on the build that has the
gather: `cache_dot` spends a `vgatherdps` per eight values to save three
quarters of the cache traffic, on a build 0.8.1 showed is not against the memory
in the first place. If that holds on a quiet host, the gather is the wrong shape
for this and the byte cache wants a different read path rather than a faster
one. Which is a measurement, and `TODO.md` carries it.

### Known gaps

What the byte cache costs in decode rate is untested, and 0.8.1's table cannot
be extended with it from here — see above for why. It wants the desktop those
numbers were taken on, and it wants both builds, because the AVX2 gather and the
four scalar reads SSE2 falls back on will not answer the same way.

The divergence measurement is still text only, and still greedy. What a
quantized cache costs a sampled run at length — where the draw is already
stochastic and a shifted distribution may not be visible at all — is a different
question and an open one.

---

## 0.8.4 — one broadcast instead of two, and three questions answered

### Scope

`TODO.md` put six items on the shipped export. Three of them were questions
rather than work — is the gain mirror worth its bytes, is the gather the wrong
read for the byte cache, and where does a picture's time actually go — and all
three are answered here, on a host that holds still. One of the three turned
into a change; two turned into an answer of "no", which is worth having written
down at the same weight.

The change that came out of it is two lines of the two bit decode kernel and
does not move a single bit of any result on the shipped export.

### A different host, and why it is the right one for this

Every number before this release was taken on a four core 2017 desktop. This
one is a four core virtual machine, and it is not that desktop: it is slower at
arithmetic and about as fast at memory, which puts decode much further from the
wall than 0.8.1 measured. The read only sweep of a 2.4 GB buffer, three passes,
best kept, is 0.8.1's own:

| threads | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- |
| sequential read | 10.41 GiB/s | 19.82 | 21.79 | **27.24** |

Against that, before any change here:

| build, four threads | decode | which is | of what the memory gives |
| --- | --- | --- | --- |
| tuned, AVX2 | 6.33 tok/s | 4.73 GiB/s | 17% |
| default, SSE2 | 5.45 tok/s | 4.07 GiB/s | 15% |

The desktop was at 42% and 30%. So this host is a poor place to ask a memory
question and a good place to ask an instruction one, which is what the three
open questions are. It is also quiet, which 0.8.3's host was not: four adjacent
tuned runs on the same prompt gave 6.51, 6.43, 6.37 and 6.32 tok/s, a spread of
3% where 0.8.3 saw a factor of two.

### The gain mirror has nothing to convert on this export

Two items — the candidate under "spend fewer instructions" and "cache
dequantized scales for the hottest planes" — were the same trade seen from two
directions: a per-plane float mirror of the group gains, memory against
conversions. Both are closed, and neither by a measurement: the export does not
give them anything to do.

Walking every plane the token loop reads, and counting what a product against
each streams:

| what a step sweeps | |
| --- | --- |
| two bit codes | 366.00 MiB |
| four bit codes | 335.25 MiB |
| eight bit codes | 26.25 MiB |
| gains | 4.59 MiB |

and the gains are already what the mirror would hold. All 548 `weight_scale`
tensors in the checkpoint are `F32` of shape `[rows, 1]`: one gain per row, in
the dtype the kernel wants. `plane_gain` is therefore a plain indexed float
load, taken once per row beside a dot product 768 to 12288 elements long —
1,203,456 of them a token against 727.5 MiB of codes, which is one load per 634
bytes read. There is no conversion to save and the mirror would be a verbatim
copy of 4.59 MiB.

One tensor in the export does carry more than one gain a row —
`embed_tokens_per_layer.embedding_scale`, `F32` of `[262144, 35]` — and it is an
indexed table rather than a swept plane. A step reads one row of it, 0.005 MiB,
and those gains are `F32` too.

This is a fact about the export rather than about the idea. A checkpoint that
shipped `bf16` group scales would put the trade back on the table, and 0.1.0's
note about scales staying in their stored dtype is still the right rule for one.
On this one there is nothing to trade.

### One broadcast instead of two

What was left of "spend fewer instructions in the decode kernels" is the two bit
path, and it is more than half of what a step sweeps: 366 MiB of the 727.5.

The AVX2 path read a dword of codes and broadcast each half of it separately,
then shifted each broadcast by 0, 2, 4 … 14 places and masked two bits off. A
variable shift reaches bit thirty-one, so the upper eight codes are the same
broadcast shifted by sixteen places more, and the mask keeps the same two bits
either way. One broadcast per sixteen codes rather than two.

The lanes, the codes and the order of the sum are untouched, so this is the same
number to the last bit. On the export's widest two bit row, 12288 by 1536, one
thread: 0.0027 s becomes 0.0019 s. `kern_code_spread` beside it takes the same
change, which is where prefill's share of it comes from.

Two other candidates were measured on the same row and not taken. Building the
float from the code's bits — `0x4B000000 | code` read as a float, less 2^23,
which is exact for codes this small and keeps the conversion off the shuffle
port — came in at 0.0021 s, behind the broadcast it would replace. The same
trick on the four bit path made it worse: 0.0011 s becomes 0.0014 s.

Decode on the shipped export, four cores, tuned:

| threads | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- |
| before | 2.63 tok/s | 4.45 | 5.75 | 6.54 |
| after | 3.19 | 5.39 | 6.86 | **7.13** |
| gain | 21% | 21% | 19% | 9% |

Over three adjacent pairs at four threads on a shorter run the same change reads
6.33 to 7.23 tok/s, and prefill on a 721 id prompt 11.16 to 11.48. The default
build does not have this path and does not move: 5.59 against 5.62.

The gain narrowing as threads are added is the shape of a build walking towards
the memory rather than away from it. At four threads the tuned build is now
reading 5.33 GiB/s of the 27.24 the sweep gives, so there is a great deal of
room left and the next thing to spend it on is no longer the two bit decode.

### The byte cache: the gather is the wrong read, measured

0.8.3 could not time `--cache 8` — its host decoded the same build in the same
configuration at 4.52 tok/s in one window and 2.32 in another — and left three
tuned pairs of one sign as "not a measurement". This host answers it. Six pairs,
each pair adjacent, on a 721 id prompt at four threads, 64 tokens:

| build | float | bytes | |
| --- | --- | --- | --- |
| tuned, AVX2 | 5.77 | 3.84 | |
| tuned, AVX2 | 5.63 | 3.84 | |
| tuned, AVX2 | 5.54 | 3.79 | −33% |
| default, SSE2 | 4.75 | 4.11 | |
| default, SSE2 | 4.56 | 4.21 | |
| default, SSE2 | 4.71 | 4.16 | −11% |

Every pair puts the byte cache behind, and the two builds are behind by
different amounts — which is the whole point, because the two builds differ in
exactly one thing: `cache_dot` reads its table with a `vgatherdps` on AVX2 and
with four scalar loads on SSE2. The build with the gather loses a third; the
build without it loses a ninth.

The clearest way to say it is that with the byte cache the tuned build is slower
than the default build — 3.82 against 4.16 — having been a fifth faster than it
with floats. A wider kernel that reads its operands through a gather is not a
wider kernel.

So the trade is confirmed as 0.8.3 guessed it: the byte cache saves three
quarters of the cache traffic on a build that was never against the memory, and
pays for it with a gather per eight values. What it collects — 1803.0 MiB of
cache down to 450.8 — is unchanged and is still the reason to have it. It is a
footprint option, not a speed one, and the flag stays off by default.

What a different read should be is now a better question than it was, and there
is a larger answer in it than a faster table read. This export ships
`num_key_value_heads` of one against eight attention heads, so `group_share` is
eight and every one of a layer's eight heads scores against the *same* cached
key row and blends the *same* cached value row. `session_layer` decodes each of
them once per head: eight table reads of every byte, where one would do. A read
that decoded a block of rows once and then ran all eight heads over the floats
would cut the decode work by eight without touching the storage saving, and it
would serve every backend rather than only the one with the gather. That is a
restructuring of the attention loop rather than a kernel swap, and `TODO.md`
carries it in that shape now.

### The towers: the loops the TODO named are not where the time is

0.7.2 left "the score and blend loops are scalar where the dot product has a
vector path". They are scalar in the source. They were not scalar in the
binary — the compiler had been vectorizing them — and writing them out through
the macro layer buys almost nothing.

An image at the full patch budget is a 48 by 48 grid, 2304 patches, sixteen
layers of twelve heads. Timed inside one thread, of the 148 s that whole run
takes: the score loop is 9.6 s, the softmax over the scores 6.6 s, the blend
7.9 s and the projection out of attention 2.6 s. Twenty-four seconds of a
hundred and forty-eight. The rest is the projections and the feed-forward, which
is where the six hundred billion multiply-adds actually are, and they already
run through the packed kernels.

Two of the three changes were still worth making and are kept, because both are
free:

- **The blend is blocked by value.** `kern_blend_rows` holds thirty-two running
  sums in registers across every span rather than folding each span into the
  destination, which is one memory operation per multiply-add instead of three.
  A value still takes its spans in the order it took them. It is worth about 2%
  of a picture on the default build and nothing measurable on the tuned one.
- **The conformer's depthwise kernel is turned tap-major once, at bind time.**
  The checkpoint stores it one row per channel, which made a tap a stride of the
  whole state and the innermost loop a walk of the kernel per channel per frame.
  Tap-major it is `kern_fma_row`, an element-wise multiply-add over every
  channel at once. A channel still sums its taps in the order it did.

The third was measured and dropped. Vectorizing the conformer's score — the sum
of a query against a key plus the lag projection — is a reassociation, because
the loop it replaces is a serial reduction that no compiler may reorder. It
measured at nothing on the shipped export, the conformer's window being thirteen
keys wide, and it moved the tower's logits in the third decimal. A refactor
should not be spending that, and 0.8.3 said so about a different one.

### The result moves nothing

Everything in this release is held to the build before it, on the shipped
export, through `logits` rather than through a greedy continuation: text, a
picture at the full patch budget, a thirty second clip, and a picture and a clip
in one prompt, on both the tuned and the default builds. Every distribution is
byte for byte what it was. The suite's `shot` section, which holds three prompts
through the whole stack to what the judged build produced, passes on both.

That is stated as a result rather than as an aspiration because the first draft
of the blend did not have it. Written with a fused multiply-add — which the loop
it replaced was not getting — it changed the greedy continuation of a picture at
the seventh token. The kernel adds and multiplies the way each build's loop did,
and the check above is what caught it.

### Testing

Two assertions, and both are held against the loop they replace rather than
against a tolerance. `kern_blend_rows` is compared value for value with the
row-at-a-time blend over five value counts that straddle its thirty-two wide
block, four span counts, and both strides it is called with — the head width
where rows are heads of a wider array, and the head size where they have been
gathered into a run. 390 assertions, up from 389, passing on the scalar, SSE2
and AVX2 backends.

### Known gaps

The 2017 desktop's table in `README.md` is left as it was. This release's gains
are measured on a different machine and the desktop's percentages of memory do
not transfer; what does transfer is that the two bit path spends one broadcast
where it spent two, which is not a property of a host.

The softmax over a picture's scores is 6.6 s of a 148 s single threaded run and
is a scalar `expf` per patch pair per head per layer — a billion of them for one
picture. Nothing here touches it.

`run.py check` and `run.py parity` were not re-run: the machine this was
measured on has no `transformers`, and every distribution the engine produces on
the shipped export is byte for byte what the previous build produced, so a
comparison against the reference would be comparing the same numbers it compared
before. That is an argument, not a run, and it is worth saying which.

---

## 0.8.5 — the cache read once for the heads that share it, the pictures, the turns after the first, the prompt that need not be primed twice, and the odd widths

### Scope

`TODO.md` put one item at the head of each of its two groups. The first was the
restructuring 0.8.4 arrived at while measuring the byte cache: eight heads read
the same cached row and the loop read it eight times. The second was the reader
that was missing — most pictures a caller actually has are jpeg, and `--image`
refused all of them. The four items under that one — the range of png the reader
would not take, the single turn the CLI would not go past, the long prompt it
primed again every run, and the bit widths that had no vector path — are done
here too.

The first moves no bit of any result and takes back almost all of what the byte
cache cost. The second is a new decoder of about four hundred lines, and the
third turns out to be one loop rather than two features; both have an encoder of
their own beside them in the tests. The fourth needed one function in the engine
and the rest in the front end, the fifth three, and the sixth is arithmetic
about where a byte boundary falls rather than a kernel at all.

### Reading a cached row once for all the heads that share it

This export ships `num_key_value_heads` of one against eight attention heads, so
`group_share` is eight: every head of a layer scores against the same cached key
row and blends the same cached value row. `session_attend` walked head by head,
which reads every cached byte eight times and, on a byte cache, decodes it eight
times.

`session_attend_group` walks the other way round. A run of cached rows is laid
into `cache_room` once — `CACHE_BLOCK_BYTES` of floats, small enough to stay in
the first level cache — and then every head of the group reads it there. On a
byte cache that is one table lookup a row instead of one a row a head. On a
float cache it is one sweep of the store instead of eight.

What it costs is a row of scores per head of the group rather than one, because
a group is scored before any of it is softmaxed. That is `score_stride` in the
session's rooms: a few megabytes beside a cache measured in gigabytes.

**It is the same arithmetic.** The block hands each head exactly the floats
`cache_dot` would have looked up for it, and the dot and the blend it then runs
are `kern_dot_real` and the blend loop the float cache always used — which
0.8.3 had already made term for term identical to `cache_dot` and `cache_add`.
The `logits` distribution and forty-eight greedy tokens of `chat` are byte
identical to 0.8.4's on both builds and at both cache settings.

Six hundred and seventy-nine ids, sixty-four tokens, four threads, three
adjacent pairs a configuration on the quiet virtual machine 0.8.4 used:

| build, cache | decode before | after | | prefill before | after |
| --- | --- | --- | --- | --- | --- |
| tuned, bytes | 5.91 tok/s | **7.22** | +22% | 14.85 tok/s | **18.94** | +28% |
| tuned, floats | 6.96 | **7.32** | +5% | 19.52 | **19.69** | +1% |
| default, bytes | 5.18 | **5.87** | +13% | 10.22 | **11.77** | +15% |
| default, floats | 5.70 | **6.03** | +6% | 11.82 | **12.15** | +3% |

Every pair is of one sign and the three runs of a configuration never overlap
the three of its pair.

The headline is the first row against the second. 0.8.4 measured the byte cache
a third behind the float cache on the tuned build and a ninth behind on the
default, and said the flag was a footprint option rather than a speed one. On
this host that gap was 15% and 9%; it is now 1.4% and 2.7%. **`--cache 8` is
now very nearly free**, and it still takes the cache at full span from 1803.0
MiB to 450.8. It stays off by default because of what it costs in accuracy —
greedy decoding still diverges at the first genuinely close call — and not any
more because of what it costs in time.

The float cache gaining 5% is the part that was not asked for. The block was
written for the table lookup and it turns out the sweep is worth something too:
eight heads reading a shared key plane of hundreds of kilobytes read it out of
the second or third level cache eight times, where the block reads it once and
is read out of the first. So `session_attend_wide` does not ask which storage
the layer is on. It asks only whether more than one head shares a row, because
where a head has its own there is nothing to divide and the block would be a
copy for its own sake.

Prefill gains more than decode on the byte cache because a prefill lane attends
over the whole run behind it, so the rows are more of what it does.

### Reading jpeg

`image_read` sniffed three magics. It sniffs four now, and `FF D8 FF` reaches a
baseline decoder: the marker walk, a canonical Huffman decode per component,
dequantization against the tables `DQT` carried, an eight by eight inverse
cosine transform, chroma upsampling at whatever the sampling factors say, and
YCbCr to RGB. Restart markers are stepped over, and a scan of one component is
walked as that component's own blocks, so a file that sends its planes one after
another reads as well as an interleaved one.

Refused, each in the shape the png reader refuses interlacing: progressive
(`SOF2`), lossless, arithmetic coded and hierarchical frames, four component
files — CMYK and YCCK need an inversion rule this reader cannot check — and a
precision other than eight. Progressive is the one that will be missed, and it
is a second decoder rather than a fourth branch of this one: coefficients
arriving across several scans with successive approximation need the whole
picture's coefficients held until the last scan lands.

Three decisions worth naming.

**The transform is float and rounds once.** Two eight by eight products through
an orthonormal basis built once per picture, and a single round to nearest at
the level shift. A fixed point transform would round twice and land within a
step of this.

**Chroma is blended, not repeated.** A component below the peak sampling is
sampled at the picture's pixel centres and interpolated linearly between the two
nearest samples in each axis. On the two-to-one factors every real file uses,
that is exactly the three-quarters-and-a-quarter blend libjpeg calls fancy
upsampling. Repetition would cost the same and put a step on every chroma edge.

**A marker inside the entropy stream feeds zeros rather than failing.** An
encoder ends a run on a byte boundary and a decoder that needed the last few
bits of a block would otherwise refuse files every other reader accepts. What
keeps that from swallowing a truncated file is that the scan refuses one that is
still fabricating bits with more than its last unit to go.

The canonical decoder beside deflate's was not widened to serve both, which
`TODO.md` had floated. `DHT` ships the table in exactly the form `puff_tree`
holds — a count per code length, then the symbols in code order — so there is
nothing to build; and the walk over it reads bits the other way round, most
significant first, out of a stream where `FF 00` means a literal `FF`. What
could have been shared is four lines of arithmetic over a bit reader that could
not be.

#### What it is checked against

`test_jpeg` carries a baseline encoder of its own: its own forward transform in
double precision, its own canonical code assignment, its own bit writer, and the
coefficient order derived from the diagonals of the block rather than copied
from the reader's table. It quantizes with tables of ones, so a round trip loses
only what the two transforms round. Its Huffman table is deliberately not the
specification's — eight to twelve bits over all 256 symbols, an incomplete code
no encoder in the wild produces — so the reader's walk is exercised rather than
a table it might have been written around.

Six pictures have to come back as the picture that went in: grey, three
component, chroma at half the horizontal sampling, chroma at half the sampling
in both directions, a file broken by a restart marker after every unit, and one
held constant over each eight by eight tile. The last two are the ones with
teeth. The tile constant file has nothing in it but dc coefficients, so the
transform round trip is exact and the colour transform is the only thing between
the samples and the pixels — half a level is the floor there, which catches a
coefficient a percent wrong or a pair the wrong way round. The plane file has a
chroma pair that is linear in both directions, where averaging a block gives its
centre and interpolating between centres gives the plane back exactly, so a
reader that repeated the nearest sample instead is caught by a floor a real file
could not be held to. A progressive frame header and a truncated entropy stream
both have to be refused.

Every one of those floors was checked by breaking the reader on purpose: the
colour coefficients moved by a percent, the two chroma bands swapped, the
vertical blend dropped, the horizontal blend dropped, the restart predictors
left unreset, and two entries of the coefficient order transposed. Each fault
fails the test that is meant to catch it, and the first pass of the suite caught
none of them, which is why the floors are where they are now rather than where
they started.

Against libjpeg, which is the comparison that matters and is not in the suite
because it is not in the repository: seven files written by Pillow at four
qualities, three sampling factors, with and without restart markers and with and
without optimized Huffman tables. Grey agrees to within a single level. Colour
agrees to within 2.8 levels of 255 at the worst pixel and 0.4 on average, which
is the two decoders' rounding and their transforms, not a disagreement about
what the file says. Odd sizes down to one pixel by one, an 800 by 600, a CMYK
file and a progressive file all do what they should. Four hundred mutations of
a real file — bytes flipped, streams truncated — produce no fault under the
address and undefined behaviour sanitizers, and neither does the suite.

#### The question the harness was going to have to answer

`TODO.md` asked which way `app_diff.py` should handle a jpeg, given that the
harness opens the picture with libjpeg where the engine would open it with its
own transform: feed both sides the same decoded pixels, or hold the jpeg cases
to a looser floor and say so.

Neither, as it turns out, because the harness already does the first. `diff_tower`
feeds the reference `tower_seed_rows` — the normalized patches the engine says it
read, out of the activation dump — and the seam's graph half is fed the same. The
only thing either half reads from the picture file itself is its width and height,
in `seam_reference_count`, and the two decoders agree about those exactly. So the
decoder is not in the comparison at all, and a jpeg case is held to the same floor
as a png one without any change to the harness. The item is closed rather than
carried.


### Reading the rest of png

The png reader took eight and sixteen bit samples and non-interlaced files. It
takes every depth the format defines now — grey at one, two, four, eight and
sixteen, palette at one through eight, RGB and the two alpha forms at eight and
sixteen — and it reads interlaced files.

The two turned out to be one change. An interlaced file is seven lattices, each
a picture of its own in the stream: its own rows, its own filter byte a row, and
its filters looking back only within the lattice. A file that is not interlaced
is the same walk with one lattice that catches every pixel. So `png_read` grew a
pass loop with one shape rather than two paths, `png_pass_bytes` rounds a row up
to the byte where a depth packs more than one sample into one, and `png_sample`
reads a sample at whatever the depth packs it — two bytes big endian at sixteen,
one at eight, and the high bits of a byte before the low ones below that. The
filter still walks whole bytes with a step of one where a pixel is narrower than
a byte, which is what the format says.

The one place a lattice is not simply a smaller picture is where it catches no
pixel at all. A lattice with rows but no columns contributes nothing to the
stream, and counting it as a filter byte a row makes the reader expect more
bytes than the file carries. That is a picture narrower than the lattice — three
pixels across, or one — and it is the case `test_png_wide` writes on purpose.

`test_png_wide` carries a writer of its own: its own chunk framing and check
values, its own line filters applied forward from the definitions, its own bit
packing, and a deflate stream of stored blocks, which is a compressor the test
does not need to have. Every depth of grey and of palette, interlaced and not,
the wider kinds interlaced, a picture smaller than the lattice and a picture of
one pixel — twenty-six fixtures, each held to the samples it was built from
exactly rather than to a tolerance, because nothing in a png is lossy.

The empty lattice is the reason those fixtures exist. Pillow does not write
interlaced png at all — it accepts the flag and ignores it — so the hundred and
ninety-two files this reader was first checked against libpng with were every
one of them non-interlaced, and the bug lived through all of them. It failed on
the first two fixtures the suite wrote. The cross-check runs the other way round
now: the suite's own fixtures are read back by libpng, thirteen of them
interlaced, and the two readers agree on all of them.


### More than one turn, and more than one conversation

The `chat` task took one turn and closed the session. `chat --loop` keeps it
open and reads more turns from standard input, and holds up to sixteen
conversations on the one loaded model.

The engine needed one thing for it. `token_frame_parts` frames a first turn:
the document's opening, then the user's turn, then the opening the model answers
into. A turn that follows one the model has already answered needs the same
frame with a different beginning — the id that closed the model's turn, which
the sampler stopped on and never fed back, because a session's cache already
holds everything before it. That is `token_frame_next`, and it is
`token_frame_inner` with a flag rather than a second framer: past the close, a
later turn is the first turn without its opening, which is what `test_turn`
asserts by comparing the two arrays.

What makes a loop a conversation rather than a series of prompts is that the
cache carries. `test_turn` states that as an equality: a session fed two turns
in two calls reaches, to the last bit, what a session fed the whole transcript
in one call reaches. It also opens a third conversation, feeds it something
else, and requires the first to reach exactly where it did before — which is the
property the split between `app_model` and `app_session` exists for, and which
no test stated until now.

The loop's own commands are `/image` and `/audio` — a picture or a clip in front
of the next turn, which makes the multi-modal path reachable mid-conversation
rather than only from the command line — `/new`, `/talk n`, `/list`, `/drop`,
`/help` and `/quit`. A turn goes through `main_reel_build` whichever turn it is,
so a later turn carries attachments exactly the way the first one does.

`/list` reports what each conversation holds because it is worth knowing: a
conversation costs a cache at the window's full span, which on the shipped
export at the default window is 1803 MiB and at `--window 4096` is 67. Opening
one that will not fit says so and leaves the loop where it was, and so does a
turn that will not fit the window — `session_prime_media` checks before it
consumes a single id, so the conversation is exactly where it was and `/new` or
`/drop` is the way on.

The conversations take their turns one at a time, and `TODO.md` carries the
reason as a task of its own: the sessions are independent, but every kernel
underneath them reaches the model's one `pool_group`, which is a fork and join
with no queue in it. Two sessions stepping at once would be two callers inside
that fork. A pool a session owns or a queue in front of the one pool are the two
answers, and neither is worth guessing at without a caller that needs it.

The single turn path is untouched: `logits` and forty-eight greedy tokens of
`chat` on the shipped export are byte identical to what they were, at both cache
settings.


### A prompt need not be primed twice

`session_save` writes a session's cache to a file and `session_load` reads it
back, and `--keep <path>` is the front end over them. On the shipped export a
687 id prompt takes 39 seconds to prime and 2.7 seconds to read back, and the
answer that follows is the same to the byte.

What goes in the file is what the session actually holds: the ids it was fed,
and the rows of each layer's key and value cache that carry anything. A layer
sized for a hundred and thirty thousand positions and holding six hundred writes
six hundred, so the file is the size of the prompt rather than of the window —
21 MiB for that prompt, and 5 MiB with `--cache 8`, which is the same quarter
the byte cache collects in memory. A ring that has turned over writes its whole
span, because every slot of one is live.

The file is host native: the same floats and the same bytes the cache holds, in
the order the machine holds them, as the mapped checkpoint is. It is a thing to
keep beside a run rather than a thing to send anywhere. `keep_mark` mixes every
shape the layout depends on — the layer count, the head counts, the window, the
cache storage, each layer's span and width — with the checkpoint's own mapped
size, and a file that disagrees with it is refused rather than restored.

Two things the ids cannot say for themselves.

**A picture is not its placeholders.** Two different pictures lay down the same
run of placeholder ids, so a cache matched on ids alone would be restored for
the wrong picture. `session_save` takes a `stamp_value` from the caller and
hands it back unread; the front end puts a hash of the embedding rows the towers
made there. Running the same prompt with a png and then with a jpeg of the same
picture is the case: same ids, different rows, and the second run primes from
nothing.

**A prefix is not a match.** The file is reused where its ids *begin* the prompt
about to run, and the difference is primed onto it. What that does not stretch
to is a prefix shorter than the file, because a sliding layer's ring cannot be
trimmed back: it holds the last `slide_span` rows written, and the rows a
shorter prompt would need behind them are the ones it overwrote.

The file holds a prompt rather than a conversation, and `TODO.md` carries the
difference as its own item. It is written before the first token is sampled, so
a rerun starts where the last run started — which also means the session that
wrote it is one id short of the prompt, the id `session_step` was about to be
fed, and a turn framed onto it would drop that id.

`test_keep` states the point as an equality: a session that reads the file
reaches the same logits as the session that wrote it, bit for bit, without
priming a single id. It also requires the ids, the peaks and the stamp to come
back, and requires a truncated file, a file that is not one of these, a file
whose mark disagrees and a file that is not there each to be refused with the
session left cleared rather than half fed.

### The odd bit widths stop walking the bit stream

Two, four and eight bits had vector paths in the fused dot and in
`kern_code_spread` beside it. Three, five, six and seven read one code at a time
through `pack_read`, which walks the stream per code, and they were between
twenty and thirty times slower than the widths beside them.

What they have in common is arithmetic rather than a kernel: at three, five, six
and seven bits, eight codes occupy exactly `bit_count` bytes. So a run of eight
always begins where the run before it ended, on a byte boundary, and a block of
eight is one word of at most seven bytes and eight shifts. `kern_code_word`
assembles that word a byte at a time — the packing is defined by the bit stream,
and reading it as an integer would be defined by the host's byte order, which is
the rule `pack_read` already follows — and `kern_code_eight` shifts the eight
codes out of it.

Best of seven on a row of 12288, one thread, in codes a second:

| width | dot, AVX2 | | dot, SSE2 | | spread, AVX2 | |
| --- | --- | --- | --- | --- | --- | --- |
| 3 bit | 0.68 G → **5.57** | 8.2× | 0.85 → **1.57** | 1.9× | 0.72 → **2.43** | 3.4× |
| 5 bit | 0.58 → **4.29** | 7.4× | 0.75 → **1.55** | 2.1× | 0.62 → **2.22** | 3.6× |
| 6 bit | 0.58 → **3.76** | 6.5× | 0.74 → **1.43** | 1.9× | 0.61 → **2.06** | 3.4× |
| 7 bit | 0.50 → **3.55** | 7.1× | 0.67 → **1.21** | 1.8× | 0.53 → **2.00** | 3.8× |

Two and four bits are untouched and measure untouched: 15.93 against 15.96 and
15.80 against 15.82.

The two builds take different paths and the measurement is why. Writing the
block to scratch and reading it back as one thirty-two byte vector is eight four
byte stores feeding one wide load, which is a store forwarding stall, and it
costs nearly half the loop: 0.94 G codes a second against 3.97 on a five bit
row. AVX2 can avoid the scratch entirely — `_mm256_srlv_epi64` shifts each lane
by its own amount, so the eight codes come out of the word inside the vector —
and it does. SSE2 has no variable shift, and reading the scratch back as a
vector there was measured at nothing over the bit stream walk, where reading it
back a value at a time is worth twice: 1.62 against 0.75. So everything that is
not AVX2 takes the scalar read, and that includes NEON, which has the variable
shift AVX2 uses and might do better still with it — unmeasured, on no host here,
so it is not guessed at.

The shipped export has no rows at these widths and not a bit of it moves. What
this is for is a checkpoint that does, and `test_kernel` now holds the dot and
the spread against the bit stream at every width the format allows, at spans
that end mid-block, at leads that are and are not where a block begins, and with
the code flip on and off — four hundred and forty-eight cases a width where
there were six.

### Everything else

The engine end to end on the shipped export, one scene written as a png, as a
4:4:4 jpeg and as a 4:2:0 jpeg: three descriptions of the same building, sky,
sun and grass. `--image` takes jpeg everywhere it takes png, including in the
media parity workflows, and `chat --loop` will take one mid-conversation and
answer questions about it two turns later out of the cache.

The suite is 460 tests from 390 — 477 before the packed kernel's twenty-four
separate assertions became seven that cover four hundred and forty-eight cases
each — clean on the scalar, SSE2 and AVX2 backends and under the address and
undefined behaviour sanitizers. Every floor the new tests hold to was checked by
breaking the thing it covers on purpose.

---

## 0.8.6 — jpeg the rest of the way, the softmax without a call, a conversation that keeps, and the odd widths off the byte

### Scope

Four items off `TODO.md`, one of them the head of each group.

The head of "anywhere" was progressive jpeg, which is what a browser is served a
fair share of and what this reader refused. It arrives here with the two smaller
gaps named beside it — a four component file, and twelve bits a sample — so
there is no jpeg this reader turns away now but one that wants another entropy
coder.

The head of "on the shipped export" was the picture attention softmax, a scalar
`expf` a patch pair a head a layer and the largest single thing in a picture that
is not a matrix product. It is a series now, eight lanes at a time.

Beside them, a session file that says whether it holds a prompt or a
conversation — so the loop can put a conversation down and pick it up in a later
process — and the last thing the odd bit widths were still paying for, which was
a word assembled a byte at a time.

### Progressive jpeg

**A second decoder, not a fourth branch.** A sequential block is final when its
scan has read it, so it is dequantized and transformed where it is read and no
coefficient buffer outlives it. A progressive block arrives across several scans
with successive approximation: a band of coefficients at a time, and a bit plane
at a time within a band. So the frame's whole coefficient store is held in
`coef_data` until the last scan lands, and the transform is one pass over it at
the end.

What the two share is the whole of the rest — the marker walk, the canonical
Huffman decode, the bit reader with its `FF 00`, the eight by eight transform,
the upsampling and the colour transform, all unchanged. What is progressive's
alone is the store, the four scan kinds — dc first, dc refine, ac first, ac
refine — and the end-of-band run, which is a code standing for a run of whole
blocks rather than a run of zeros inside one.

**The refining scan is the one with the shape.** In a first scan of a band the
run field of a code counts zeros, as it does in a sequential block. In a
refining scan it counts only the coefficients the earlier scans left zero: every
coefficient they left nonzero carries one correction bit wherever the walk
passes it, and those bits follow the code rather than leading it. A coefficient
a refinement makes nonzero has a magnitude of exactly one, which is why the size
field is always one there and the bit after the code is only its sign.

**A component's quantization table is claimed when its first scan names it.**
A progressive component is dequantized at the end of the file rather than where
it is read, so which table stood at that moment is the one that has to be kept
rather than whichever `DQT` was last seen. A component no scan ever names has no
coefficients and no claim, and the frame is incomplete rather than grey.

**Four bands are ink.** `APP14` is read now, which is the only thing in a file
that says whether three bands are a luma and a chroma pair or the picture's own,
and whether four are CMYK or YCCK. Four bands are turned first where the marker
says YCCK, then multiplied by the black — which is what a file written inverted
means, and every writer that ships the marker inverts. Four bands with no marker
to read are still refused: nothing in the pixels says which four they are, and
guessing would be a picture rather than a decode. Three bands with no marker are
a luma and a chroma pair unless the component ids are `R`, `G` and `B`.

**Twelve bits a sample** widens the level shift, the plane and the tables. The
plane is `uint16_t` and the coefficient store `int32_t`, so the same code serves
both depths and the sequential path came along with it.

**The tests grew an encoder.** `test_jpeg` had a baseline encoder of its own;
it now has a progressive one beside it, over a scan script that reaches all four
scan kinds and splits a band across two scans besides — the dc plane sent a bit
short and then refined, the luma's low frequencies and its high ones as separate
first scans two bits short, then a refinement of each band in turn. The same
pictures have to come back through it subsampled, at one component, at twelve
bits and broken by restarts.

Beyond that, the reader was held against libjpeg over 72 generated files — grey
and colour, progressive and sequential, three subsampling factors, four sizes,
three qualities — and agrees within four levels of 255 everywhere, within one on
CMYK. That is a cross-check rather than a test in the suite, because the suite
does not have libjpeg and should not need it.

### The softmax exponential

**The call is the cost.** At the full patch budget a picture's attention is
about a billion exponentials — a patch pair, a head, a layer — and `expf` was
being called for each. Nothing about them is hard: a softmax has already taken
its own peak off every value, so every argument is at most zero and the total is
at least one.

**So it is the definition instead.** `e^x` is `2^k` times `e^r` where `k` is the
whole number nearest `x / ln 2`, which leaves `r` no further from zero than half
of `ln 2`. Over that range the series for `e^r` is finished after eight terms:
the ninth is five parts in a thousand million of the value, an eighth of the
last bit a float carries. `2^k` is an exponent field written straight into the
word. `ln 2` comes off in two pieces, the first exact in a float, because in one
piece `k * ln2` rounds away the low bits of `x` that the remainder is made of.
The coefficients are the reciprocals of the factorials the definition names
rather than a fitted set, so there is nothing there to have copied wrong.

**Held against libm**, the worst relative error over the whole range a softmax
can reach is 1.19e-07 — one unit in the last place — and a whole softmax of 2304
against a double precision reference is within 6.7e-07. The kernel goes from 3.2
ns a value to 0.64 on the tuned build and 1.70 on the default one, and its three
passes — the peak, the exponentials with their total, the scaling — each go a
lane at a time on all three vector paths, with the scalar helper as the tail and
as the fourth path.

**On the shipped export** a picture at the full patch budget, single threaded,
goes from 122.8 s to 113.2 s.

**It moves the numbers, and here is the honest measure of how much.** Take the
old kernel, keep `expf` exactly, and accumulate the total in two accumulators
instead of one — a pure reassociation, a last bit either way. Over the top 128
of a picture prompt that moves the logits by 0.46 of a logit on average and
drops fifteen of the hundred and twenty-eight; this change moves them 0.53 and
drops twelve. A picture is amplified that way because 35 routed layers stand
behind it and a near tie in an expert routing flips. The argmax is the same
token in every comparison run, and a text only prompt is unchanged to the digit
the `logits` task prints.

### A conversation, not only a prompt

`--keep` writes its file before the first token is sampled, which is what makes
it useful — a rerun of the same prompt starts where the last run started — and
what makes it not a conversation: the session that wrote it is one id short of
the prompt it holds, the id `session_step` was about to be fed, and a turn
framed onto it would drop that id.

The cache itself is the same either way, so nothing in it can tell the two
apart. What tells them apart is now a word in the file. `session_save` takes it
and `session_load` hands it back; the mark text carries a version, so a file
written before this is refused rather than read as a prompt it might not be.
`--keep` writes and reuses only a prompt.

The loop gains `/save <path>` and `/open <path>` over the other one. A
conversation is written after an answer, holds every id, and is picked up by
framing the next turn onto it — so a later process carries on from where an
earlier one stopped and the model remembers what it said. The turns it holds
ride in the caller's stamp, which the engine still never reads: a conversation
restored whole has nothing to match a stamp against, and whether the model has
already answered is what decides how the next turn is framed. A prompt offered
to `/open` is refused with the reason rather than continued.

On the shipped export: a turn, a `/save`, and in a fresh process an `/open` and
a follow-up question, answered out of what the first process said.

### The odd bit widths

0.8.5 put three, five, six and seven bits on a block of eight codes — a byte
boundary at every one of those widths — but assembled the block's word a byte at
a time, because the packing is defined by a bit stream and reading it as an
integer is defined by the host's byte order. That was the last thing the width
still decided.

**So the host is stated rather than worked around.** This engine is
little-endian by decision: `real_read` casts mapped bytes to a `float` or a
`uint16_t`, and the code stream is a dense little-endian bit field the kernels
index directly, so a big-endian host would not read the weights wrongly in one
kernel but in every one of them. It is said once, beside `pack_read`, and the
compiler is asked to say so where it knows.

**With that, a block is one load.** The eight bytes are the word already, and
the bits above the block's own are never reached by a shift the block takes. The
exception is the tail, which is why `run_end` is carried: a block at the end of
a row has as few as three bytes behind it, and the five past them may be past the
end of the mapping rather than merely past the end of the row. There the word is
assembled as it always was, which is a handful of blocks a row against the
thousands that are not.

Best of interleaved runs on a 12288 row, one thread, G codes a second, with two
and four bits carried as untouched controls:

| path | 3 bits | 5 bits | 6 bits | 7 bits | 2 bits | 4 bits |
| --- | --- | --- | --- | --- | --- | --- |
| tuned dot | 4.47→6.05 | 3.73→6.04 | 3.34→6.09 | 3.14→5.93 | 16.13→15.61 | 15.99→15.15 |
| tuned spread | 2.40→2.42 | 2.15→2.49 | 2.03→2.59 | 1.95→2.50 | | |
| default dot | 1.37→1.60 | 1.32→1.57 | 1.24→1.60 | 1.20→1.56 | 10.70→11.13 | 7.76→7.72 |
| default spread | 2.27→2.47 | 2.16→2.47 | 2.00→2.46 | 1.94→2.43 | | |

The plainest reading of it is not the percentages but that all four widths now
land on the same rate where they did not: the width has stopped shaping the
loop. Against two and four bits the fused dot is 38% of them on the tuned build
rather than a fifth to a quarter, and what is still missing is a vector path for
the spread at these widths.

The bound on the wide read is held by a test rather than by padding: a row
allocated to exactly the bytes it packs into, over every width and every whole
block of it. It is `malloc`'s row rather than the engine's, because the engine's
allocator rounds every block up to the alignment its kernels want and a row
asked for at twenty-four bytes is sixty-four. With the bound taken out, the
sanitizer build fails on that case, which is what says the test has teeth.

The shipped export still has no rows at these widths and not a bit of it moves.

### Everything else

The suite is 492 tests from 460, clean on the default, tuned and sanitizer
builds. The engine end to end on the shipped export reads a progressive jpeg
through the vision tower and describes it.

---

## 0.8.7 — the odd widths off the scratch, and a tier above AVX2

### Scope

Two items off `TODO.md`, one from each group.

From "anywhere", the spread's vector path at three, five, six and seven bits —
which is what 0.8.6 named as the whole of the distance still between those
widths and the two the checkpoint leans on. It is closed, and closed for the
fused dot at the same time, because both readers now decode a block the same
way.

From the same group, the AVX-512 path beside AVX2. The item asked for evidence
before a wider kernel rather than an assumption, and this is the first host to
carry both the instructions and the headroom to show it: a bare sweep of memory
gives 33.99 GiB/s at four threads here and the tuned build reads 5.25 of them,
so the decode kernels are spending instructions rather than waiting.

The evidence also said where to stop, which is the more useful half of it. Three
kernels went wide and two were written, measured and taken out again.

Every rate below is the best of four interleaved runs on one host, 0.8.6's build
and 0.8.7's two measured in the same loop. They are not 0.8.6's numbers and do
not compare with them: this host is roughly half of that one on the paths both
tables carry.

### The odd widths, decoded in one shuffle

**A block of eight always begins on a byte boundary, so every block of a width
picks the same bytes at the same shifts.** That is the property 0.8.6
established and did not spend. Eight codes are exactly `bit_count` bytes, so
the offsets within a block are the width's own and not the block's: they can be
built once for a run and read by every block in it.

So the decode is one `shuffle_epi8` over a sixteen byte load. Each lane is
handed the four bytes its code starts in — a code of seven bits or fewer
straddles two of them at most, so four is more than enough — and shifts its own
code down and masks it. The bytes a block reaches are all within the first ten,
so the low half of the load is broadcast to both halves and the one shuffle
serves all eight lanes despite picking across what would otherwise be a lane
boundary.

**The bound is asked once for a run rather than once a block.** A block at the
end of a row has as few as three bytes of the row behind it and the sixteen byte
load would reach past the mapping rather than merely past the row.
`kern_code_wide_span` says how many codes have the bytes, and the run is split
into two loops there: the blocks that have them take the shuffle, and the
handful at the end take the broadcast word 0.8.6's fused dot used. Asking it per
block instead is a branch inside the loop and was measured: it costs the spread
nearly all of the win, 3.6 G codes a second rather than 8.9. That is the whole
reason it is a second loop rather than an `if`.

**Both readers go through it.** The fused dot and the spread had drifted into
unpacking a block two different ways — one inside the vector, one through
scratch — and there was never a reason for two. `kern_code_wide` is the one
decode and `kern_code_eight` is now defined only on the hosts that still read a
block through scratch.

On a 12288 row, one thread, tuned, in G codes a second, with two, four and eight
bits carried as untouched controls:

| path | 3 bits | 5 bits | 6 bits | 7 bits | 2 bits | 4 bits | 8 bits |
| --- | --- | --- | --- | --- | --- | --- | --- |
| spread | 1.22→9.17 | 1.21→9.08 | 1.22→9.37 | 1.21→9.24 | 10.33→10.31 | 8.79→10.76 | 9.48→9.49 |
| fused dot | 3.36→6.38 | 3.36→6.45 | 3.37→6.39 | 3.36→6.32 | 11.00→10.92 | 8.46→8.56 | 10.78→10.87 |

The spread is between seven and eight times what it was and within a tenth of
the two widths the checkpoint packs, which is the item closed rather than
narrowed. The fused dot is nearly twice what it was, at 59% of the two bit rate
where 0.8.6 left it at 38%. The controls are flat but for the four bit spread,
which nothing here touched and which moved 22% on code layout alone — worth
saying, because it is the size of a change one might otherwise claim.

The shipped export packs no odd width, so not a bit of it moves. The bound is
held by 0.8.6's test extended to every row length from eight codes to two
hundred rather than one, each row allocated to exactly the bytes it packs into:
the split between the two loops lands at a different block for every width and
every length, and the sanitizer's guard sits behind the last one. With the bound
claiming one block more than the row can carry, that case stops and the one
before it does not.

### A tier above AVX2

**AVX-512 is a tier, not an alternative.** A host with AVX-512 has AVX2, so the
macro layer sets both names rather than choosing between them, and only the
kernels with something to gain from sixteen lanes are written twice. Everything
else compiles as the AVX2 path it always was. `back_flavor` answers the wide
name first, or a wide build would report itself as the tier it stands on.

The four subsets asked for are `f`, `bw`, `dq` and `vl` — the ones the kernels
reach for — which keeps the path off the early parts that have `f` alone.
`run.py --wide` selects it and implies `--tuned`.

**The fused dot is what went wide, at the three widths the export packs.** Four
bits: sixteen packed bytes are thirty-two codes, so the nibble split is done
once over twice the bytes. Eight bits: thirty-two bytes are thirty-two codes,
flipped in one exclusive or. Two bits: a dword is sixteen codes, which is one
whole vector, so the two halves the AVX2 path shifts out separately become one
shift. On the same row and the same runs, in G codes a second:

| path | 2 bits | 4 bits | 8 bits |
| --- | --- | --- | --- |
| fused dot, tuned | 10.92 | 8.56 | 10.87 |
| fused dot, wide | 16.07 | 13.18 | 14.77 |

**Two kernels were written wide and taken out again, and that is the finding.**
The spread was written for sixteen lanes at two and four bits and is quicker
measured on its own — 13.16 G codes a second at two bits against 10.19, and
12.34 at four against 9.98 — and it made the engine slower: prefill on the
shipped export fell 13% while decode, which never reaches the spread, rose. A
spread is a small part of what prefill does and the per-lane sums it feeds are
the rest; a five hundred and twelve bit store in the middle of them costs those
sums more than the wider store saves. A kernel 29% quicker in isolation that
leaves the program 13% slower is the whole argument for measuring the program.

The eight bit spread went the same way for a plainer reason: a byte is a code,
so there is nothing to unpack and the loop is a load, a convert and a store,
which a compiler vectorizes well on its own. Written out by hand it reaches 9.88
G codes a second against the 11.13 the compiler's own loop gets under the same
flags.

**`-mprefer-vector-width=256` is part of the flag set for the same reason.**
Letting the compiler widen its own vectorization costs both halves: decode 3.84
tok/s against 4.12 and prefill 4.15 against 4.39. The kernels written for
sixteen lanes still get them; nothing else does.

**What it is worth on the shipped export**, best of three interleaved runs:

| | tuned | wide | |
| --- | --- | --- | --- |
| prefill, one thread | 5.00–5.09 | 5.37–5.40 | +7% |
| decode, one thread | 3.17–3.18 | 3.85–3.86 | +21% |
| prefill, four threads | 15.22–15.75 | 15.44–16.22 | +3% |
| decode, four threads | 7.13–7.21 | 7.55–7.63 | +6% |

Both halves gain and decode gains most, which is what a wide fused dot should
do. The four thread column is the smaller one because the memory is more of the
cost once four threads pull on it — 5.6 GiB/s of the available 33.99 — and that
gap between the two columns is the reading the TODO's head item asked for rather
than a disappointment.

**Sixteen lanes reassociate the sum.** On "The capital of France is" the top
logit is 27.111 on the default build, 27.250 on the wide one and 27.473 on the
tuned one: the new tier lands between the two that already ship, and the token
is the same. Cross-backend bit-identity was never a property here — the tuned
build has an FMA the default one has not — and this does not make it a smaller
one.

### Everything else

The suite is 493 tests from 492, clean on the default, tuned, wide and scalar
backends, and under the address and undefined sanitizers on the tuned and wide
ones. The reference comparison passes on the shipped export with no check
failed, on the default build and on the wide one — the second of which it could
not do before this release, because `check` rebuilt the engine without the
flags it was given and compared a default binary whatever was asked for. On the
wide build the largest logit gap of the four prompts is 1.74 against a bar of
2.24, which is twice how far the reference moves against itself.
