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

---

## 0.8.8 — what four threads were actually waiting on, and where a picture goes

### Scope

Two items off `TODO.md`. The head one is what the four bit and eight bit decode
paths wait on once four threads pull on them; the other is the fresh profile of
a picture that "speed up the towers further" asked for before anything else is
tried there.

On the first: 0.8.7 could say that a wide fused dot was worth 21% of decode at
one thread and 6% at four, and could say that the memory was more of the cost at
four than at one, but it could not say how much more. That was the question, and
it is answered here. The answer is that at four threads the kernels were not the
cost and neither was the memory. Between the two there was a third thing nobody
had measured, and it was a third of the token: the fork and the join around
every projection.

On the second: the profile is here, and it moved the question from the tower to
the batch that every tower and every prefill shares. Three candidates for that
batch were measured and refused — a wider dot, a deeper chain, more lanes — and
the fourth was taken: four lanes now share the row's load rather than each
loading it again, which is prefill 23% quicker on the default build and a
picture's projections 8% quicker, without moving a bit of any result.

### The host

Four cores of a virtual machine, an Intel Xeon at 2.8 GHz with AVX-512, a
mebibyte of L2 a core, 33 MiB of L3 between them and 15 GiB of memory. The
weights are the shipped export, mapped and warm in the page cache. A read only
sweep of a 2.4 GiB buffer, three passes, best of four:

| threads | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- |
| sequential read | 9.38 GiB/s | 18.70 | 20.97 | **26.27** |

That is 0.8.4's host almost exactly at four threads — 26.27 against 27.24 — and
rather slower at one.

### The kernels, against a working set that fits and one that does not

The same `kern_dot_code` over the same 12288 wide rows, twice: once over half a
mebibyte of codes read four thousand times, which never leaves the cache, and
once over two gibibytes read once, which never stays in it. The gap between the
two is what the memory costs the loop; the resident column is what the loop
costs by itself. In GiB/s of codes read, best of four:

| build | width | resident, 1 | streaming, 1 | resident, 4 | streaming, 4 |
| --- | --- | --- | --- | --- | --- |
| tuned | 2 bits | 3.00 | 2.81 | 11.63 | 10.98 |
| tuned | 4 bits | 4.80 | 4.17 | 17.99 | 14.86 |
| tuned | 8 bits | 12.06 | 7.38 | 47.45 | 23.95 |
| wide | 2 bits | 5.16 | 4.37 | 19.97 | 15.61 |
| wide | 4 bits | 8.16 | 5.64 | 31.94 | 21.69 |
| wide | 8 bits | 18.23 | 8.39 | 72.17 | 27.66 |

Read across, at four threads: the eight bit path keeps half of its resident rate
and lands at 23.95 GiB/s on the tuned build against the 26.27 the sweep gives,
and at 27.66 on the wide one, which is the sweep's own number handed back. That
path waits on the memory and has for a while. The four bit path keeps 83% of its
resident rate on the tuned build and 68% on the wide one: the memory is part of
what it waits on and not all of it. The two bit path keeps 94% and 78% — it is
still spending instructions, which is what 0.8.4 said of it and what the wide
build's 42% gain over the tuned one at that width says again.

So each path on its own has an answer, and it is the one the item guessed at.
What the item did not ask, and what turns out to matter more, is whether those
three answers add up to the engine.

### They do at one thread and they do not at four

A step sweeps 366 MiB of two bit codes, 335.25 of four bit and 26.25 of eight —
0.8.4's census, unchanged. Through the streaming rates above, at one thread on
the wide build, that is 143 ms of kernel. The engine's token was 177 ms. So the
kernels are 81% of a single threaded token, which is what a profile says
independently: on an `-O2 -pg` build of the same source, 80% of decode is inside
`kern_mat_vec_band`, where `kern_row_code` and the fused dot are inlined.

At four threads the same arithmetic gives 38.9 ms of kernel. The engine's token
was 95.1 ms. The kernels are 41% of it, and the other 56 ms are somewhere the
memory is not, because the memory hands the same bytes to the same loop in
38.9 ms when it is asked for them without stopping.

### It is the fork and the join, and there are 277 of them a token

Every projection is a `pool_run`: the caller publishes a task, broadcasts,
takes a band itself, and then waits for the workers to count themselves done.
Counted on the shipped export, decode issues **277 of them a token**, and not
one of them takes the single threaded path the 64 row bar keeps small planes on.

A fork and a join of a task that does nothing, on this host:

| threads | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- |
| before | — | 23.38 us | 59.77 | **119.95** |

At four threads that is 33 ms a token of publishing and waiting, against the
56 ms the kernels do not account for. The rest is the attention, the norms, the
sampler, and bands that do not divide evenly — but the fork and the join are
the larger half of it, and they were nobody's kernel.

The shape of the cost is the scheduler rather than the mutex. A worker that has
finished its band sleeps on a condition variable; the next projection is tens of
microseconds later; waking it is a trip through the kernel on a core that has
three other threads' work queued behind it. Four threads cost more than two
because four wakes contend where two do not.

### Spin before sleeping, where every thread has a core

Both waits — the worker's for a task and the caller's for the workers — now read
the counter they are waiting on directly for a bounded spell before they take
the lock and sleep on it. Nothing else changes: the counter is a hint, the lock
is still what orders the memory either side of a job, and a waiter that spins
past its bound sleeps exactly as it did before.

| threads | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- |
| before | — | 23.38 us | 59.77 | 119.95 |
| after | — | **0.71** | **3.82** | **17.39** |

Which on the shipped export is, best of three interleaved runs a cell,
`bench --serve 16` on a seven id prompt:

| build | | 1 thread | 2 | 3 | 4 |
| --- | --- | --- | --- | --- | --- |
| wide | decode, before | 5.32 tok/s | 8.84 | 9.92 | 10.51 |
| wide | decode, after | 5.66 | 9.44 | **12.61** | **15.69** |
| wide | prefill, before | 6.29 | 11.39 | 15.07 | 18.13 |
| wide | prefill, after | 6.44 | 11.51 | 15.81 | **19.63** |
| tuned | decode, before | 4.04 | 6.23 | 8.07 | 8.37 |
| tuned | decode, after | 4.18 | 7.06 | **9.62** | **11.21** |
| tuned | prefill, before | 5.92 | 10.69 | 14.91 | 17.43 |
| tuned | prefill, after | 5.97 | 10.87 | 15.29 | 18.42 |
| default | decode, before | 2.27 | 4.02 | 5.43 | 6.19 |
| default | decode, after | 2.24 | 4.25 | 5.97 | **7.26** |
| default | prefill, before | 3.15 | 5.66 | 8.31 | 10.32 |
| default | prefill, after | 3.12 | 5.86 | 8.32 | 9.95 |

Decode at four threads is 49% quicker on the wide build, 34% on the tuned one
and 17% on the default one, and the order of those three is the argument itself:
the quicker the kernel, the larger the share the waiting held. At one thread
there is no pool and no fork, and the three rows say so.

Prefill gains far less because it does not fork nearly as often: sixteen tokens
go through a projection in one job, so a batch of ids pays one fork where
sixteen decode steps pay sixteen.

The scaling is the plainest reading. From one thread to four the memory hands
over 2.8 times as much, and decode on the wide build produced 1.98 times as many
tokens before this and 2.77 times as many after.

### A spin is only right where the core is spare

Eight threads on four cores decode 6.88 tokens a second, and spinning made that
3.95. The core a spinner holds is exactly the one a worker with real work on it
is waiting for, so the wait is no longer a wait for the other side but a delay
to it.

So `pool_open` asks whether every thread of the pool has a core of its own and
gives the group a spin of zero where they do not — which is the behaviour this
had before, sleeping on the first look. Re-measured at eight threads on four
cores: 6.68 before, 7.01 after, which is the spin gone rather than the spin
helping. The default thread count is the host's core count, so the common case
spins and a deliberately oversubscribed pool does not.

### The picture, profiled again

The other item this release touches asked for a fresh reading of where a
picture's time goes, because the one it carried was 0.8.4's and the softmax line
of it had since been removed. Here it is, on the same host, single threaded, on
the wide build, at the export's full patch budget — a 48 by 48 grid, 2304
patches, sixteen layers of twelve heads — with a timer around each part rather
than a sampling profiler, because `gprof` divides a callee's time among its
callers by call count and every call here is a different size:

| part of the tower | 0.8.4, its host | 0.8.8, this host |
| --- | --- | --- |
| the projections | not separated | **24.0 s** |
| the scoring | 9.6 s | 6.5 |
| the softmax | 6.6 | **0.83** |
| the blend | 7.9 | 5.0 |
| everything else in it | | 3.3 |
| the tower, end to end | | 39.1 s |

Three things in that. The softmax line is gone, which is 0.8.6's series arriving
where it was aimed: eight tenths of a second where the call cost six and a half.
The projections are 61% of the tower and the attention 32%, so the balance
0.8.4 described — attention conspicuous, projections the rest — has tipped
further towards the projections than it names. And the scoring and the blend are
what they were, within the difference between two hosts.

The third thing is not in the tower at all. A picture at this budget lays down
256 soft tokens, and those are prefilled through the text stack like any other
ids: 40.1 s of it, against the tower's 39.1. **Half of what a picture costs is
the text stack reading what the tower produced**, which no reading of this
before had separated. At four threads the two are 13.8 s and 13.6 s.

### Three ways to hurry the projections, measured and not taken, and the fourth

The projections are `kern_row_code_many`: a group of codes spread into a float
scratch once, then dotted against each of sixteen lanes through `kern_dot_real`,
which is the single hottest function in the engine — 65% of the self time of a
picture's whole run. Three candidates, each quicker where it was measured on its
own and none of them quicker in the engine:

**A sixteen lane `kern_dot_real`.** On the 256 element span the batch hands it,
27.08 G multiply-adds a second against 17.97 for the eight lane path it would
replace. Put the spread that fills its row in the same loop, which is where it
actually sits, and that narrows to 20.64 against 16.71. In the engine it is
worth 5% of the tower's projections — 22.79 s against 23.88 — and costs prefill
2%, 17.08 tokens a second against 17.49 on an 1800 id prompt. A wash on one half
and a loss on the other, so it is not in.

**Four eight lane accumulators instead of two.** The multiply-add is four cycles
deep and the host retires two a cycle, so two chains leave the units half fed:
24.88 G multiply-adds a second against 17.94, a 39% gain measured on the span
the batch uses. In the engine, prefill on the tuned build is 15.10 tokens a
second against 16.66 — 9% slower — which is the same shape of answer 0.8.7 got
from the wide spread and worth writing down twice.

**Thirty-two lanes a batch rather than sixteen**, which halves the number of
passes a prompt makes over the codes and halves the spreads with it: prefill
17.29 against 17.31, which is nothing, and sixty-four lanes 16.11, which is
worse. So the codes are not what the batch is waiting on either.

Between them those three say what the next attempt should not be. The batch is
not waiting on the width of its multiply-add, nor on the depth of its dependency
chain, nor on the codes it streams. What is left is the shape of the loop, and
that is the last change in this release.

### The batch shares the row's load

**Two loads for every multiply-add is the limit, and one of the two is the same
load sixteen times.** The batch spreads a group of codes into a scratch and then
dots that scratch against each of sixteen lanes, a lane at a time. Every lane
reads the whole scratch again. The host will issue two loads and two
multiply-adds a cycle, so a loop that needs two loads per multiply-add gets one
multiply-add a cycle whatever its vector is — which is exactly why a wider
vector, a deeper chain and more lanes a batch all left it where they found it.

**Four lanes at a time share it.** `kern_dot_real_many` holds four lanes'
accumulators at once and loads the row once for the four: ten loads for eight
multiply-adds rather than sixteen. On the 256 element span the batch hands it,
with the lanes strided as the batch strides them, 25.93 G multiply-adds a second
against 17.11 for a lane at a time; at the wider stride of the feed-forward,
20.32 against 14.89.

**Not a bit of any result moves, and that is a property rather than a hope.**
Each lane keeps the two accumulators `kern_dot_real` keeps, takes the same slots
into the same one in the same order, ends with the same reduction — factored out
now as `kern_dot_total`, so the two cannot drift — and folds the span's scalar
tail in before the total reaches the caller's accumulator, which is where the
one lane path folds it. Lanes past the last block of four, and every path
without a vector, go through `kern_dot_real` itself. A test walks every lane
count from one to nine against every span from one to forty and requires the
same float, not a near one; and the whole distribution over the vocabulary is
byte for byte identical on the default, tuned and wide builds over three
prompts.

**What it is worth**, best of two runs on an 1800 id prompt at four threads:

| build | prefill before | after | |
| --- | --- | --- | --- |
| default, SSE2 | 9.28 tok/s | **11.40** | +23% |
| tuned, AVX2 | 16.75 | **17.80** | +6% |
| wide, AVX-512 | 17.75 | **18.41** | +4% |

The order is the argument again, and it runs the other way from the pool's. The
default build has no fused multiply-add: a multiply and an add for every element,
against the same two loads, so the loads are the largest share of what it does
and halving them is worth most. The wide build has the most arithmetic per load
already and gains least.

Single threaded on the same host, the wide build's prefill goes 6.76 to 7.07
tokens a second, and a picture's projections — the 24 s this release measured —
go to 21.90, which takes the tower from 39.50 s to 37.61.

**Four lanes and not eight**, though eight measures quicker: 30.13 G
multiply-adds a second against 27.15. Eight lanes with two accumulators apiece
is seventeen vectors live and the host has sixteen, so the quicker form is the
one with a single accumulator a lane — which reassociates every sum in the
engine for about two per cent of a token. Bit-exactness is worth more than that.

The obvious answer to the register count was measured and is not one. A host
with the wide tier has thirty-two vector registers, which is room for eight
lanes and both accumulators and therefore for an eight lane block that is still
exact; written that way it reaches 24.04 G multiply-adds a second, behind the
four lane form's 27.15. So what the eight lane block gains, it gains from having
half as many accumulators to keep and not from the lanes, and there is no
version of it that is both quicker and the same arithmetic.

### Not a bit of any result moves

Neither change moves a number. The pool hands the same slices to the same
workers in the same order and the task functions are untouched; what changed
there is how a thread waits between two jobs. The batch's blocking is held to
the same floor by construction and by a test, as above.

The suite is 498 tests from 493, clean on the default, tuned and wide builds and
under the address and undefined sanitizers on the default and wide ones. The
synthetic sweep is unchanged and still passes: eleven configurations, every
tensor of every layer. The reference comparison passes on the shipped export on
the wide build, with the same logits it reached before.

### What is left of the four thread question

At four threads the wide build now reads 761 MiB a token at 15.69 tokens a
second, which is 11.65 GiB/s of the 26.27 the sweep gives — 44%, where before
this it was 30%. The kernel floor for that mix is 38.9 ms and the token is
63.7 ms, so about 25 ms a token is still outside the kernels: 4.8 of it is the
277 forks and joins that remain, and the rest is the attention, the norms, the
sampler and the bands.

The next reading of the head item is therefore two questions rather than one.
The kernels themselves want the eight bit path left alone — it is against the
memory and has been measured there — and the two bit path, which is more than
half of what a step sweeps and still spending instructions. And the 25 ms
outside them want the same treatment this release gave the fork: measured
rather than guessed at.

---

## 0.8.9 — the product on the grid the export already rounds to

### Scope

The head item on `TODO.md`, taken by the first of the two routes `RESEARCH.md`
named for it — its idea 2, the integer matrix kernel — rather than by another
pass over the float one.

The item was scoped as a two bit kernel, because two bits is the width where
the float path was still spending most of its time on instructions rather than
on memory. What the route turns out to give is all three widths the export
packs, both halves of the engine, and the memory as the only thing left in any
of them. At four threads the two bit path now reads 25.90 GiB/s of codes where
the bare sweep gives 32.18, the four bit path 29.47 and the eight bit path
31.04. Before this the same three were 12.26, 17.37 and 23.17.

On the shipped export, at four threads on the host below: decode 8.92 tokens a
second to 12.57, prefill 15.19 to 33.96, and a picture from 36.3 seconds to
20.7. At one thread, where the memory is further away, prefill is 5.21 to 16.86
— three and a quarter times.

It moves the numbers, and where the numbers can be held against something that
is not another float summation it moves them the right way. Against the
reference's own tower modules on the shipped weights, the vision tower goes from
2.871 off to 2.049, where the reference moves 2.945 against itself; the audio
tower from 7.739 to 7.086 against its own 7.213. The float sum was the
approximation and the integer one is exact.

### What the export already guarantees, and what nobody was using

`quantization_config` in this checkpoint gives 548 planes an
`input_activation_scale`, and `plane_lift_many` has always applied it: before
any code plane's product, `quant_step` rounds every activation onto that step,
clamps it to the eight bit range, and writes back `level * step`.

So the activations entering a code plane are not arbitrary floats. Each one is
an integer between -128 and 127 times a scalar the plane carries. The sum the
kernel wants is

    Σ code · act  =  step · Σ code · level

and the right hand side is an exact integer, of an integer code and an integer
level, both of which fit in a byte.

The float path spent its instructions undoing that. It unpacked each code,
converted it to a float, and multiplied it by an activation that was itself an
integer wearing a float's clothes — three vector operations and a
multiply-add for every sixteen codes at two bits, and the same again for the
next sixteen. The integer path does not undo it: one `vpdpbusd` takes
sixty-four codes against sixty-four levels and adds their products into sixteen
`int32` lanes, and the whole block costs one load, one variable shift, one mask
and that instruction.

The eight bit levels also shrink the other side of the loop. A row of 12288
columns read the activation vector as 48 kilobytes of floats and now reads it as
12 kilobytes of bytes — first level cache traffic that a row pays every time,
against a staging that the product pays once.

### The activations are reordered, not the codes

Sixty-four codes come out of the packing in one broadcast. Two bits: sixteen
packed bytes are broadcast to the four quarters of a vector, each quarter
shifted down by its own bit position within a byte, and masked. Four bits:
thirty-two bytes to two halves, shifted by zero and four. Eight bits: the bytes
are the codes and only the flip is taken. A shift of a thirty-two bit lane moves
every byte inside it by the same places, which is why one variable shift serves
a whole block whatever the width.

What that leaves is a block whose codes are in the order *bit position first*:
at two bits, byte `16r + k` of the vector holds the code of column `4k + r`. The
obvious repair is a shuffle to put them back. That shuffle would be paid by
every row of the plane — 262144 of them for the output head.

So the codes are left where they land and the activations are written where the
codes will be. `kern_level_stage` walks a block and writes level `4k + r` into
slot `16r + k`, which is a permutation of a vector that is read once and costs
nothing measurable against the rows that read it. Sixty-four is one block for
every width, because two, four and eight all divide a byte.

The columns past the last whole block — a plane of 194 columns has two — are
staged where they lie, because there is no block for them to be ordered by.

Everything else has to be read in the staged order, and the scalar path is the
part of that which is easy to get wrong: it was written to walk the columns and
read the levels beside them, which is right for the tail and wrong for every
whole block. On the host this was developed on the vector path consumes every
whole block and the mistake is unreachable, so the unit tests passed. Built for
SSE2, AVX2 and AVX-512 without `vnni` — where the scalar loop is the whole
kernel — twenty-four of them failed. The scalar loop now walks the same
`run_wide * part + run` to `part_count * run + part` map the staging wrote, so
the two paths read the same bytes in the same pairs, and it is the reference the
vector path is held against on every tier rather than only on the one that
skips it.

### Two things are checked rather than assumed

The kernel does not trust the caller to have rounded. `kern_level_of`
recomputes `level * step` in single precision and compares it to the activation
it came from: where `quant_step` wrote the value this is the same product
rounded the same way, so it is exact by construction, and where it is not, the
comparison says so. One activation off the step refuses the staging for the
whole product and it falls back to the float path — every lane of one call
answered the same way, so a batch is never half on the grid.

The accumulator is checked too. `kern_level_ready` holds `span · 255 · 128`
against `INT32_MAX` before the path is taken, which is the whole group's sum
against what one of the sixteen lanes that share it can carry — deliberately
sixteen times stricter than it needs to be, because the slack is free. On this
export the widest group is 12288 columns, so the bound it is held to is 401
million against 2147, and every plane passes; the check is written for a plane
that would not.

The correction is integers as well. A group's whole term is
`Σ code·level − (bias + zero) · Σ level`, and both sums and the zero point are
exact integers, so the difference is taken in `int64` and only the two gains are
floats. What used to be a chain of a thousand rounded products is now two
roundings.

### `vnni` is asked of the host rather than of the tier

`APP_SIMD_AVX512VNNI` is its own name beside `APP_SIMD_AVX512`, and this is not
tidiness. The four subsets 0.8.7 asked for arrived together; `vnni` arrived two
generations later, so a machine with all four may still not have it and a build
that assumed it would stop with an illegal instruction rather than fall back.
`run.py --wide` asks the compiler what `-march=native` would define on the build
host and adds `-mavx512vnni` only where the answer says so. Where it does not,
`kern_level_ready` returns zero, the staging never runs, and the float kernels
are the whole of the engine.

That is checkable and was checked: on this host a `--tuned` build and a
`--wide` build with the flag withheld both produce the shipped export's `logits`
output byte for byte identical to 0.8.8's. Only the build that has the
instruction changes, and it changes only by being exact.

### The host

The same machine as 0.8.8: four cores of a virtual machine, an Intel Xeon at
2.8 GHz with AVX-512 and VNNI, a mebibyte of L2 a core, 33 MiB of L3 between
them and 15 GiB of memory. A read only sweep of a 2.34 GiB buffer, best of four:

| threads | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- |
| sequential read | 9.62 GiB/s | 18.21 | 25.29 | **32.18** |

That is faster than the 26.27 the same sweep gave in 0.8.8, on the same host and
the same code. It is a virtual machine and its neighbours are not this
repository's; every pair of numbers below was taken in one sitting against its
own control, and none of them should be read against 0.8.8's column.

### The kernels, measured the way 0.8.8 measured them

The same `kern_dot_code` over the same 12288 wide rows, against
`kern_dot_level` over the same rows and the staged levels of the same
activations: once over half a mebibyte of codes read five hundred times, which
never leaves the cache, and once over two gibibytes read once, which never stays
in it. In GiB/s of codes read, best of three:

| width | path | resident, 1 | streaming, 1 | resident, 4 | streaming, 4 |
| --- | --- | --- | --- | --- | --- |
| 2 bits | float | 3.59 | 3.05 | 13.52 | 12.26 |
| 2 bits | **integer** | **15.36** | **7.50** | **59.92** | **25.90** |
| 4 bits | float | 6.06 | 4.70 | 21.51 | 17.37 |
| 4 bits | **integer** | **30.77** | **8.69** | **118.17** | **29.47** |
| 8 bits | float | 13.18 | 6.05 | 45.75 | 23.17 |
| 8 bits | **integer** | **68.08** | **8.31** | **229.50** | **31.04** |

Resident, the integer path is four to five times the float one at every width.
Streaming, it is not, and that is the finding rather than a disappointment: at
four threads all three widths land between 25.90 and 31.04 GiB/s against the
32.18 the sweep gives, which is the memory handing back its own number. At one
thread they land between 7.50 and 8.69 against 9.62, which is the same thing
said again with less headroom. 0.8.8 could say that of the eight bit path alone.
It is now true of all three, and the question the kernels answer is closed:
there is nothing further to win in them without reading fewer bytes.

Eight bits is the width that says the loop shape matters. Written with the
width as a runtime value, the block's `slot / codes_per_byte` is a sixty-four
bit division inside the loop, and at eight bits — where the float path is
already quick — that division cost more than the whole path it replaced: 7.44
GiB/s resident where the float path measured 13.2 in the same sitting. The loops
are written out per width instead, so the divisor is a literal and the division
is a shift, and the table above is what that is worth at every width.

### The engine, on the shipped export

A 288 id prompt, 32 tokens served, wide build, one run each taken back to
back:

| threads | prefill before | prefill after | decode before | decode after |
| --- | --- | --- | --- | --- |
| 1 | 5.21 tok/s | **16.86** | 3.97 tok/s | **6.09** |
| 2 | 9.10 | **25.98** | 5.91 | **9.06** |
| 4 | 15.19 | **33.96** | 8.92 | **12.57** |

Prefill gains more than decode, and by more the fewer threads there are. Both
follow from the same fact: a batch reads each code byte sixteen times and a
decode step reads it once, so prefill was the more instruction bound of the two
and had the more to give up. Decode at four threads is now reading 9.63 GiB/s of
the 784.4 MiB it sweeps a token, against 6.83 before.

A picture, which is both halves at once — the tower, and the 256 soft tokens it
lays down prefilled through the text stack. Wall clock of the same run with the
image and without it:

| threads | before | after |
| --- | --- | --- |
| 1 | 105.0 s | **48.3 s** |
| 4 | 36.3 s | **20.7 s** |

The tower is eight bit throughout, which is the width the item did not expect to
gain anything, and it is a batch, which is where the gain is largest.

### What moves in the numbers, and what says which way

Every result of a code plane moves, because the summation is not the float
path's. What the change is worth numerically is not a matter of opinion, because
the towers can be held against the reference's own modules on the same weights:

| tower | float path, off by | integer path, off by | the reference against itself |
| --- | --- | --- | --- |
| vision | 2.871 | **2.049** | 2.945 |
| audio | 7.739 | **7.086** | 7.213 |

Both moved towards the reference. The vision tower was inside the reference's
own movement before and is further inside it now; the audio tower was just
outside it and is now inside. That is the expected direction and the reason is
not subtle: the integer sum is exact, so the only rounding left in a row is the
two gains at the end of it, where the float sum rounded every one of a thousand
products.

The text stack does not say the same thing, and it is worth being exact about
why. `run.py check --model model --wide`, run on both builds against the same
reference on the same host:

| prompt | float gap | integer gap | the reference against itself | float top 16 | integer top 16 |
| --- | --- | --- | --- | --- | --- |
| `Hello!` | 0.8251 | 1.4239 | 0.9074 | 94% | 81% |
| `Write one sentence about the sea.` | 1.2234 | 1.3379 | 1.3961 | 94% | 81% |
| `What is the capital of France?` | 0.7999 | 0.8981 | 0.5981 | 81% | 88% |
| `Explain gravity to a child in two sentences.` | 1.7376 | 1.8306 | 1.1180 | 88% | 94% |

Both builds pass every check — the leading token matched, the greedy
continuation followed, every gap inside twice the reference's own movement,
which is the bar this repository has used since the checkpoint's activation grid
made per tensor equality meaningless. The integer build's gaps are a little
larger on all four prompts and its top sixteen share moves both ways.

That does not contradict the tower result; the two measure different distances.
A tower's rows are one pass and are compared where they come out, so "closer to
the reference" there is a statement about arithmetic and nothing else. The
logits are 35 layers away, each of which rounds its input onto the export's
grid: a
last-bit difference in a sum that lands on a half step becomes a whole step, and
a whole step at layer three is a different residual for every layer after it.
`app_diff.py` has said this since it was written — the reference in double
precision does not reproduce the reference in single by that measure either.
Being exact does not buy agreement with a float32 reference downstream of a
grid; it buys not being wrong, and the tower is where that shows.

What it also buys is that the greedy continuations are unchanged and the leading
token is the reference's on every prompt, before and after.

The three recorded prompts in `app_test.c` pass unchanged, every recorded id
still inside the sixteen ranks it is allowed to move within. That was not true
of an intermediate version of this change, and the reason is worth keeping.
With decode on the grid and prefill left on floats, the weakest of the recorded
ids — `17531` on "Say hello.", recorded at rank 7 and already at rank 11 on
0.8.8's wide build — fell to rank 17. With both halves on the same arithmetic it
sits at rank 11 again, scoring 20.039 where 0.8.8 scored it 20.014. A graph that
is exact in one half and rounded in the other is not half way between them; it
is a third thing, and it drifts further than either. That is why the batch path
is in this change rather than a later one.

### The eight lane block, settled

0.8.8 measured an eight lane batch and refused it: eight lanes are quicker only
with a single accumulator each, which reassociates the sum, and 16% of one loop
was not judged worth moving the engine's last bit. `TODO.md` carried that as an
open question rather than a closed one, because it named two larger claimants
for the same spending — the tower's attention, and an integer accumulator that
changes the arithmetic anyway — and said the bit should be moved once, for
whichever of them measured out.

This is that spending. The integer accumulator is in, it moved every code
plane's last bit, and the towers came out closer to the reference for it. The
eight lane block does not get a second hearing: the reassociation it wanted is
now spent, and what it was worth — 16% of a loop that is no longer the loop —
was never on the same scale. The tower's attention is the other claimant and is
still open.

### What was not taken

`RESEARCH.md`'s idea 3, the lookup table execution, is not here. It was the
other route named for this item and it is the one that answers the same
question — how to stop spreading codes into floats — so with the integer path
in, the case for it is a different case: it would have to beat a kernel that is
already at the memory. On this host it cannot, because nothing can. On a host
without an integer dot product it might, and that is where the experiment now
belongs.

The 256 bit `vnni` of hosts that have `avx_vnni` without AVX-512, and the ARM
dot product instructions, are both the same kernel at another width and neither
is written. This is the rule 0.8.7 set for AVX-512 and it applies to its own
successor: the path was written when a host with the instruction and the
headroom to show it turned up, and the others wait for the same.

Nothing was done to the staging's own cost. It is a scalar loop with a `rintf`,
a multiply and a compare per column, run once per product per lane — one pass
over 1536 activations against the output head's 262144 rows of them, so it is
the row count smaller than what it saves. It would be a vector loop in a day and
would not be measurable.

### The lists, rewritten around it

`TODO.md` had grown a tail of closed items and the reasoning behind them, all of
which is in this file under the version that closed it — the gain mirror and the
conformer's score loop in 0.8.4, the shared cache row in 0.8.5, the softmax
series in 0.8.6, the odd widths and the wide tier in 0.8.7, the fork and join
and the eight lane block in 0.8.8. It was duplication rather than a record, so
it is gone, and what is left in its place is a table naming the version to read.

The list it leaves is ordered by what moves a token, and it now says for each
entry whether mainline llama.cpp has the same thing — checked by reading
`ggml-org/llama.cpp` at `465e49b`. That is a prior rather than a scoreboard:
where llama.cpp has an idea the payoff is known and the work is porting, and
where it does not, either the idea is worse than it looks or this checkpoint
makes it worth more than it would be there.

`RESEARCH.md` lost its idea 2, which is this change, and lost the six ideas that
are now ordinary engineering rather than research — speculative verification,
the image budget, early token merging, the head's vocabulary bound, tiled tower
attention, and the visual feature cache — all of which moved to `TODO.md` with
what is known about them from that reading. What stays there is what still needs
a trained artifact or an untested hypothesis: the distilled encoder, the
per-layer-conditioned drafter, QAT-cell certification, structured sparsity, and
the lookup-table kernel for hosts with no integer dot product.

---

## 0.8.10 — where the other two thirds of a token actually were

### Scope

The head item on `TODO.md`, which gated the rest of the list: measure what a
token spends outside the kernels, a part at a time. It is measured now, and the
answer is not the one the item was written around — most of it was never
outside the kernels at all. Three of the four things the measurement then
pointed at were taken; the fourth was taken and reverted, and the reverting is
the interesting one.

On the reference host at four threads, on a 288 id prompt: decode 15.5 tokens a
second to 18.4, prefill 43.1 to 54.8. On a 2004 id prompt decode 13.4 to 15.5.
Not one bit of any logit moves, at any thread count, and the suite now says so
by comparing bits rather than a tolerance.

### The premise the item was written on, and why it was wrong

`TODO.md` reasoned this way. A decode step reads 784.4 MiB. The memory gives
32.18 GiB/s at four threads, so the bytes cost 24 ms. The step took 80. The
missing 56 ms had to be "somewhere the kernels are not: the attention, the
norms, the sampler, the 277 forks and joins, the bands that do not divide
evenly, the per-layer embedding lookup, the residual adds."

That list is a hypothesis about a step nobody had divided, and it is mostly
wrong. Divided, at four threads:

| part | ms a step | share | MiB a step | GiB/s |
| --- | --- | --- | --- | --- |
| mlp | 23.05 | 41.8% | 475.3 | 20.14 |
| final norm, head | 7.48 | 13.6% | 97.0 | 12.67 |
| score, softmax, blend | 4.12 | 7.5% | 26.2 | 6.23 |
| q k v | 4.02 | 7.3% | 70.1 | 17.04 |
| logit cap | 3.86 | 7.0% | — | — |
| attn out | 3.18 | 5.8% | 63.2 | 19.42 |
| ple lift | 2.72 | 4.9% | 26.3 | 9.42 |
| mlp gate | 2.57 | 4.7% | — | — |
| ple feed | 2.54 | 4.6% | 26.5 | 10.19 |
| sampler | 0.80 | 1.5% | — | — |
| norms, residuals | 0.48 | 0.9% | 1.0 | 2.07 |
| rope, cache write | 0.34 | 0.6% | 0.1 | 0.17 |
| embed, bookkeeping | 0.01 | 0.0% | — | — |

The norms and the residual adds are **0.9%**. The rotary turn and the cache
write are **0.6%**. The per-layer embedding lookup is a tenth of a millisecond
inside `ple lift`. The whole of what the item guessed at, less the attention,
comes to under 2% of a step. What the missing 56 ms was, instead, was the
kernels not going as fast in place as they go on a bench — and four specific
things sitting between them that nobody had counted because none of them reads
a weight.

This is the argument for the timer rather than for any of the changes under it.
0.8.8 found a third of a token in the fork and join by measuring; this found
the same order of thing by measuring, and would not have found it by reasoning,
because the reasoning had already been done and had produced a wrong list.

### The timer

`session_step` closes one named part as it opens the next. One clock read a
boundary and not two: the read that ends a part begins the one after it, so the
parts join edge to edge with no gap between them to lose time in, and they sum
to the step by construction rather than by hope. `bench --verbose` prints them
largest first.

Anything not marked is charged to the part that was open, which is a quiet way
to be wrong, so the parts that catch the unmarked code are named for what they
catch — `norms, residuals` inside a layer, `pass bookkeeping` outside every
layer — rather than left blank. The report closes with the two ways the claim
could still fail: `unnamed`, the step's own clock less the parts of the pass,
which measures 0.000 ms and is the accounting either side of the pass; and
`timer`, the reads the division itself spent, priced at what a read measures on
the host just after the run — 0.011 ms of a 54 ms step, 428 reads at 26 ns.

The same sweep is split a second time, into bytes, by `model_phase_bytes`, so a
part has a rate and not only a share. This is what makes the table say anything:
a part's share of the clock says which part to look at, and its share of the
bytes says whether there is anything in it to find. The suite holds the two
totals to the byte, and caught a norm missing from the split while the split was
being written — which is the whole reason the check exists, since a forgotten
sheet does not fail anything, it just makes a rate quietly wrong.

The timer is armed by `verbose_level`, so a run that did not ask for it pays a
load and a predicted branch a boundary. With it on and off, decode measures the
same within the run-to-run noise of this host.

### The attention, which was the last part running on one core

`session_attend_group` reached the pool nowhere. Every projection either side of
it was banded across four threads and the scoring between them was not, which
the table above makes obvious and which nothing before it would have. At a 4334
id prompt it was two fifths of the whole step: 38.4 ms at four threads against
39.7 at one.

It is two jobs now rather than one, because the two halves have to divide along
different axes and that is the whole design.

**The scores divide by position.** A score is one dot product against one cached
row, so a slice writes only the scores of the positions it took and reads only
the rows behind them. Nothing is summed across a slice boundary.

**The blend divides by head.** A blend is a running sum down the span, and
dividing *that* by position would regroup its additions — the same arithmetic,
a different rounding, and an engine whose answer depended on how many cores the
host has. Divided by head, each slice carries its own heads the whole way down
the span and adds in the order the serial loop added in. What it costs is that
every slice decodes the value block for itself where one slice decoded it for
all eight heads; the block is eight kilobytes and is read straight back, so what
is repeated is the decode and not the fetch.

So the export's logits are byte for byte what they were, at one, two and four
threads. The suite pins it with a `memcmp` rather than a tolerance, because a
tolerance is exactly what a regrouped sum would pass. The suite also lowers the
fork threshold, because the synthetic fixture's whole window is sixty-four
positions and at the shipped threshold the divided path would never run there —
the check would have been comparing the serial path with itself and calling it
agreement. Breaking the blend on purpose fails sixteen assertions across four
sections, including the three recorded prompts on the shipped export.

The threshold was guessed at 384 and the guess was wrong. Measured: the two jobs
add about 0.6 ms to a step — seventy forks at nine microseconds — and the phase
they divide runs 0.70 ms at a span of 45 and 6.02 ms at 288. At 288 the divided
path wins by 2.9 ms, so 384 was leaving that on the floor. The crossing was not
pinned closer than "between 45 and 288, nearer 45", so the figure is 128, the
safe side of the bracket rather than its middle.

| prompt, four threads | decode before | after | prefill before | after |
| --- | --- | --- | --- | --- |
| 288 ids | 17.8 | 19.6 | 44.5 | 46.0 |
| 2004 ids | 13.2, 13.3, 13.8 | 16.1, 16.3, 14.2 | 32.4 | 36.6 |
| 4334 ids | 10.1 | 13.0 | 25.0 | 33.2 |

At one thread nothing forks and nothing should move, and nothing does beyond the
noise — which on this host is wide enough to be worth saying: the same binary
measured 6.51 and 7.63 tokens a second on the same prompt. The four thread rows
move one way and grow with the context, which is the signal.

### The two elementwise maps between the kernels

Splitting the gelu out of the feed-forward and the logit cap out of the head
puts them at 12.9% and 6.1% of a decode step between them, both on the calling
thread with the whole pool idle. Neither had ever been looked at, and the reason
is structural: neither reads a weight, so neither appeared in any byte count,
and every profile before this one was a byte count.

Keeping them separate also fixes what the kernels either side of them appeared
to be doing. With the gelu folded in, the feed-forward reported 16.82 GiB/s;
with it counted apart, 20.14. The head reported 8.63 and reports 12.67. A phase
that mixes a code plane with a transcendental over every element quotes a rate
that belongs to neither.

**The gelu is forked and stays forked.** It is 35 jobs a step of 0.22 ms each,
and a clean win in every round: decode 17.5, 19.2, 20.1 tokens a second against
15.8, 16.3, 18.3, and prefill 51.5, 54.3, 55.9 against 42.6, 43.9, 44.5. Prefill
gains twice what decode does, because a batch runs the same gate sixteen lanes
wide and so had sixteen times as much of it to give.

**The logit cap is forked, measured, and put back.** The same reasoning applies
to it — an elementwise map, no summation order to lose — and the measurement
refuses it. Three runs each at four threads: serial 3.94, 3.82, 3.87 ms against
5.20, 1.33, 5.21 forked. It reaches the four threads' figure once in three and
is worse than serial the other two. That is the shape of a single fork whose
workers were joined on the output head a moment earlier and have not settled,
and a steady 3.87 beats a mean of 3.9 that swings by four milliseconds, because
the swing is what a caller feels and the mean is not.

The measurement is kept in the comment where the next person to have the idea
will read it, with what would actually make it pay: not a better fork, but not
calling `tanhf` 262144 times.

### What the table leaves for the next pass

The kernels are not at the memory on this export's rows, and 0.8.9's claim that
they were rests on a bench of the export's *widest* row. At four threads, with
the elementwise work counted apart: `attn out` 19.42 GiB/s, `mlp` 20.14, `q k v`
17.04, the head 12.67, `ple lift` 9.42 — against 25.90 to 31.04 on a row of
12288. The rows a decode step actually reads are 1536 wide most of the time, and
a row eight times shorter pays the per-row epilogue eight times as often. That
is the first entry on `TODO.md` now, and it is a kernel item where the list has
had none since 0.8.9 declared them finished.

`logit cap` at 7.0% and `sampler` at 1.5% are the other two, and both are the
output head's 262144 rows being touched twice more after being read. They belong
with the vocabulary bound rather than beside it.

---

## 0.8.11 — the epilogue, and two calls into libm that a step made 477184 times

### Scope

The first two speed entries on `TODO.md`. The first asked *why is a kernel
slower on the short rows a decode step actually reads than on the widest row a
bench can build*; the second asked for the logit cap without 262144 calls to
`tanhf`. Both are answered, though the first is answered differently from the
way it was asked — and looking for it turned up two more things of the same
shape that the list did not have on it at all.

On the reference host at four threads, wide build, on a 374 id prompt: prefill
**59.3 tokens a second to 82.1**, decode **21.8 to 27.1**. On a 7 id prompt
decode **23.1 to 29.3**, and the step floor 43.97 ms to 34.79.

Every number below is the minimum a phase reached over fourteen runs a side,
the two builds alternating. This host's whole-step figure swings by a fifth
between one run and the next, which is enough to invent a result and enough to
hide one; a minimum over interleaved runs survives that, because a shared host's
noise only ever adds time.

### The answer to the first entry was not arithmetic

`TODO.md` guessed the row epilogue, and named the right suspect for the wrong
reason. It supposed the cost was *the work in* the epilogue — a horizontal
reduction, a scale and a store, paid once per 384 bytes on the output head
instead of once per 3072 on a wide row — and proposed blocking the output rows
so those reductions interleave.

Read out of the generated code, the epilogue was not doing that work at all. It
was making calls. Per row: one to `kern_dot_level_rest`, which handles the
columns past the last whole block and had nothing to do because every span here
is a whole number of blocks; and one to `real_read`, the switch over every
storage type, to fetch a single `F32` gain. Around them, a `vzeroupper` and the
whole caller-saved vector state, spilled and restored in the middle of the row
loop. On a 384 byte row that is paid once per 384 bytes.

So three changes, of which the blocking is the smallest:

**A block of four output rows.** `KERN_LEVEL_ROWS_LOOP` and
`kern_row_code_level_rows`: four independent accumulators, one load of the
staged levels shared between them. The integer sum is the one row path's, bit
for bit — an integer dot does not care in what order it is taken — and the test
says so by comparing floats rather than asking for nearness.

**The remainder call guarded.** `if (slot < span_count)` before it. Where the
span is a whole number of blocks there is now no call, rather than a call that
returns zero.

**The gain read without the switch.** `plane_gain_line` picks the direct load
once per product where the scales are `F32`, which in this export they always
are, and the loop then carries a pointer rather than a branch on a type.

### And two more of the same shape

**The logit cap.** `tanh(x/c)*c` over 262144 logits, 4.45 ms of a step, on the
one thread that reaches it. 0.8.10 tried to fork it and refused with numbers.
The way through was the one that entry named and did not take: `tanh y` is
`1 - 2/(e^{2y}+1)`, and the exponential is the series `kern_exp_wide` has
carried for the softmax since 0.8.6. **4.45 ms to 0.28**, eight lanes at a time,
still on the calling thread.

It is not `tanhf` to the last bit and does not claim to be. Swept over 65536
arguments from -200 to 200 the largest gap from the call is two and a half parts
in ten million of the cap — 7.6e-6 at this export's cap of 30 — the result is
still monotone, so no sampler's choice can change except between two logits
already that close, and the ends are exact rather than near.

**The gelu.** The same call, in the other elementwise map: 6144 values a layer,
215040 a step, 2.40 ms across the whole pool while reading not one byte of
weight. Written as

```
0.5 x (1 + tanh y)  =  x t / (t + 1),      t = e^{2y}
```

it is the same exponential again, and better conditioned than the form it
replaces, because `t/(t+1)` has nothing to cancel where `1 + tanh y` loses the
low bits of a small result. Measured against the closed form in double over
200001 arguments, the series is off by at most **3.64e-7 where `tanhf` is off by
4.31e-7**: it is the more accurate of the two as well as the quicker.
**2.40 ms to 0.29.**

Under the clamp the gate is stated to be zero rather than left to the clamped
exponent. The cube overflows on the way in, so a saturated `tanh` — exactly
minus one — makes the gate exactly zero, where `x e^{-30}` at a gate of 1e30 is
9.4e16. Two instructions, and there is no argument the kernel answers absurdly.

**The bf16 dot product.** Not on the list at all, and the largest single rate
change here. `kern_dot_real` had a vector path for `F32` and a scalar loop for
`BF16` — and this export keeps `per_layer_model_projection` in bf16, 8960 rows
of 1536, **26.25 MiB that every decode step reads in full**. A bf16 is a
sixteen bit shift into the top of a word, so a block of them is one widening and
one shift; `kern_bf16_wide` is that, and nothing is rounded that was not rounded
before. **9.63 GiB/s to 23.30.**

### The step, before and after

| part | before | after | | |
| --- | --- | --- | --- | --- |
| | ms | GiB/s | ms | GiB/s |
| mlp | 17.918 | 25.90 | 16.787 | 27.65 |
| final norm, head | 6.534 | 14.50 | 6.649 | 14.25 |
| logit cap | 4.447 | — | 0.276 | — |
| q k v | 2.874 | 23.82 | 2.900 | 23.61 |
| ple lift | 2.664 | 9.63 | 1.100 | 23.30 |
| attn out | 2.400 | 25.72 | 2.299 | 26.84 |
| mlp gate | 2.399 | — | 0.286 | — |
| ple feed | 1.720 | 15.04 | 1.532 | 16.89 |
| score, softmax, blend | 1.423 | 7.60 | 1.434 | 7.54 |
| sampler | 0.828 | — | 0.784 | — |
| norms, residuals | 0.457 | 2.19 | 0.445 | 2.25 |
| rope, cache write | 0.293 | 0.20 | 0.290 | 0.20 |

### What the row block did not do, and the reframing that came out of it

`TODO.md` named the output head as the extreme case and the plane to write the
block against: 262144 rows of 384 bytes, and the slowest plane in the step by
rate. The block does not move it. **14.50 GiB/s to 14.25** — inside the noise,
which is to say unchanged — while the wide-row planes it was not written for
took most of the gain.

That is worth more than the change would have been, because chasing it turns up
the reason, and the reason is that the entry was comparing the wrong quantity.
`vpdpbusd` consumes sixty-four codes whatever their width. So a two bit plane
spends the same instruction on 16 bytes of weight that a four bit plane spends
on 32 and an eight bit plane on 64, and GiB/s is not comparable across widths at
all. Counting the multiply-adds instead, on the phase minima above:

| part | multiply-adds a step | G mac/s |
| --- | --- | --- |
| final norm, head | 402.7 M | **60.6** |
| mlp | 990.9 M | 59.0 |
| attn out | 132.5 M | 57.7 |
| q k v | 147.0 M | 50.7 |
| ple feed | 27.5 M | 18.0 |
| ple lift | 13.8 M | 12.5 |

**The output head is the fastest plane in the step, not the slowest.** It looks
slow in GiB/s because a two bit weight is half the bytes of a four bit one, and
for the same reason it is the cheapest 400 million multiply-adds the engine
performs. The two planes that are actually behind are `ple feed`, whose
`per_layer_projection` is 1536 rows of **256 columns** — four blocks and then an
epilogue, which is the short-row question in its real form — and `ple lift`,
which is a float path and always was.

### Two things measured and refused

**Eight rows a block instead of four.** Built and run: mlp 27.47 GiB/s to 27.49,
the head 14.06 to 14.08, the step floor 37.36 ms to 37.07. A wash, for eight
live accumulators and twice the code, on the same reasoning 0.8.8 refused the
eight lane block with.

**Software prefetch of the code stream.** The hardware prefetcher does not cross
a page and the head crosses one every ten rows, so one touch a block of rows,
four blocks ahead, looked free. It is not: mlp **12.7% worse**, q k v 19.5%,
ple feed 17.9%, attn out 14.0%, and nothing better anywhere. Reverted.

### What says it is still the same engine

`app_test.py` against the transformers reference passes every check on all four
prompts, on the wide build and on the plain one: rank one matches, and every
logit gap is inside the bar the reference sets against itself. On the wide build
the gaps are 1.4239, 1.3379, 0.8981 and 1.8306, which are 0.8.10's figures to
the last digit printed. Nothing here was supposed to move them and nothing did:
four decimal places do not resolve a cap that has shifted by 7.6e-6.

A note on how not to read that comparison, because this version nearly recorded
the mistake. The **plain** build reports smaller gaps than the wide one — 1.18,
1.15, 0.83, 1.20 — and set beside the wide build's older run that looks exactly
like an accuracy gain from the two series. It is not one. A plain build has no
integer dot product, so every code plane takes the float path instead, and it is
the path that moves those numbers rather than the version. Compare a build
against itself, or the comparison will tell you whatever you brought to it.

709 unit tests pass on the plain, SSE2, AVX2 and AVX-512 builds. Eight of them
are new: that a block of rows is the one row path bit for bit, that the cap is
within its stated bound of `tanhf` and never swaps two logits, that the cap and
the gate are exact at both ends, and that the gate is the closed form over a
whole row.

The NEON paths for the cap and the gate are guarded on `__aarch64__`, because
the lane divide they use is AArch64's; a 32-bit ARM host takes the scalar loop,
which is the same series. Like the MSVC paths, they have been read and not run —
there is no ARM host here.

---

## 0.8.12 — how a block of rows closes, and the plane that was paying for it

### Scope

The first speed entry on `TODO.md`: the two planes that 0.8.11's multiply-add
count found were actually behind, `ple feed` at 18.0 G multiply-adds a second
and `ple lift` at 12.5, against 50 to 61 everywhere else. The entry named two
routes for the first of them — more rows a block on narrow planes, or the group
loop lifted out so a whole plane's epilogues are one pass — and the answer is
neither exactly. It is that the rows have to close *together*.

`ple lift` is not touched. It is a bf16 float path and 12.5 G a second is what
that path gives; the entry says so, and the only thing that would move it is
quantizing the export's remaining bf16 at load, which is a footprint question
and stays under *Not speed*.

### A note on the host, because the numbers are not 0.8.11's

Every figure here was taken on a different machine from the one 0.8.11's table
came from: four cores of a Xeon at **2.1 GHz** rather than 2.8, same
instruction set, AVX-512 and VNNI, wide build. The shipped export's step floor
on it is 40.55 ms where the reference host's is 34.79, and a bare four-thread
sweep reaches 40 to 49 GiB/s where the reference host reaches 32.18. So the
absolute numbers below are **not comparable to 0.8.11's line for line** — only
the ratios are, and every ratio is a minimum a phase reached over twelve runs a
side with the two builds alternating, which is the discipline 0.8.11 set for the
same reason: a single run of this host swings by more than the whole result.

The first attempt at this change was measured against a host that had a
`pip install` running on it and reported the wide-row planes 4 to 7% slower.
They were not. It was also measured once against a page cache that no longer
held the checkpoint, and reported the change losing to the baseline. Neither
was true. Warm the cache and quiet the host, or this file records a fiction.

### Where a row's close was going

A row of the integer path ends in a horizontal sum of its accumulator, a zero
point correction, two multiplies and a store. `kern_row_code_level_rows` did
four rows at a time and closed each of them on its own: four
`_mm512_reduce_add_epi32`, four sums arriving in general registers, and four
rows of scalar arithmetic to put them back into floats.

On a 1536 column row that is one close against twenty-four blocks of dot
product and it disappears into them. On `per_layer_projection` — 1536 rows of
**256 columns**, half of what `ple feed` reads — it is one close against *four*,
and the close costs more than the dot products it closes.

0.8.11 measured eight rows a block against four and found a wash, and this
version agrees with that measurement and disagrees with what it was taken to
mean. Widening the block changes nothing on its own: eight rows of four blocks
is eight closes against thirty-two dot products exactly as four rows is four
against sixteen. The ratio is fixed by the columns. What moves it is closing
the rows *together*.

### Three changes, and the third is the one that matters

**`kern_level_fold`.** Four accumulators of sixteen lanes in, four sums side by
side in one vector out. Three rounds of pick and add put each accumulator's
four quarters into one quarter of a single register, two more sum each quarter
within itself, and one gather takes the four down to the low four lanes.
Fourteen instructions against four `_mm512_reduce_add_epi32`'s near forty — and
the sums land in a vector, which is where the next change needs them.

**`kern_row_code_level_wide`.** Four of those folds stacked into one vector of
sixteen rows, closed at once: the zero point correction is one subtract of
sixteen lanes, the sixteen gains are one load, the activation step is one
broadcast, the sixteen results are one store. The dot products are still taken
four rows at a time, four times, so no more than four accumulators are ever
live — the width is a width of the *close*, not of the loop, which is why the
register pressure that made eight rows a wash never arises.

`kern_level_wide_ready` says which planes may take it: one group a row, `F32`
gains, a whole number of blocks a row, and a span narrow enough that the
correction is exact in thirty-two bits. Every code plane in this export
qualifies. `kern_mat_vec_band` takes the wide block as far as it goes, then the
block of four, then a row at a time — three forms of one arithmetic.

**And the same fold in `kern_dot_level_many`,** which is the batched path and
was closing its four lanes with four calls for the same reason. That one was
not on the list at all and is where most of prefill's share comes from.

### The step, before and after

A 3 id prompt, four threads, wide build, the minimum each phase reached over
twelve runs a side:

| part | before | after | |
| --- | --- | --- | --- |
| mlp | 21.757 | 21.367 | -1.8% |
| final norm, head | 7.867 | 7.851 | -0.2% |
| q k v | 3.627 | 3.426 | -5.5% |
| attn out | 2.828 | 2.789 | -1.4% |
| **ple feed** | 1.577 | **1.422** | **-9.8%** |
| ple lift | 1.270 | 1.279 | +0.7% |
| step floor | 40.550 | 39.570 | -2.4% |

And on a 301 id prompt, where the batched path is reached too:

| | before | after | |
| --- | --- | --- | --- |
| prefill | 87.17 tok/s | **93.80** | **+7.6%** |
| decode | 23.42 tok/s | **24.01** | +2.5% |
| ple feed | 1.601 ms | 1.441 | -10.0% |
| q k v | 3.539 ms | 3.297 | -6.8% |
| step floor | 42.700 ms | 41.650 | -2.5% |

Decode on the short prompt is 24.66 tokens a second to **25.27**.

### What the entry asked for, and what is left of it

`ple feed` is 17.5 G multiply-adds a second to **19.4**. That is a real move and
it is not the entry closed: the plane is still a third of the rate of the four
that are not behind, and the entry's own estimate of the ceiling — about 5% of a
token for the two planes together — is larger than the 2.4% taken here.

What the next attempt should look at is not the kernel, and it was measured
rather than guessed at. `ple feed` is two pool forks a layer, seventy a step,
for two planes of 384 KiB each. Timed against the engine's own `pool_group` on
this host, 20000 rounds at four threads, **an empty fork and join is 2.95 us**,
so those seventy are **0.21 ms of the phase's 1.42** — a seventh, paid whatever
the rows cost. The same run puts a 384 KiB read at 21.9 us across the pool
against 18.6 us of work, which is the same seventh seen from the other side, and
the step's 277 forks at 0.82 ms of 39.5.

The entry stays open with that named, and `TODO.md` says what the fix would have
to be: on the spinning path a fork and join still takes about eight mutex
acquisitions and four condition broadcasts, none of which a spinning worker
needs. That is a rewrite of the pool rather than of a kernel, and it wants a
stress test against the sleeping path before it ships — which is why it is not
in this version.

### Two folds measured against each other

The first form of `kern_level_fold` reduced each accumulator to four lanes with
extracts and then combined the four with three `_mm_hadd_epi32`. The second is
the one that shipped: `_mm512_shuffle_i32x4` three times over, then two in-lane
swaps and a gather. Built and run against each other on the same twelve
alternating rounds: `ple feed` 1.398 ms against 1.422, step floor 40.010 against
39.570, decode 24.99 against 25.27. A wash on the plane it was written for, the
shuffle form ahead on the step, and it is the shorter of the two — so it ships,
and the numbers are here rather than a claim that it is faster.

### The page walk, tested and not found

`TODO.md`'s second speed entry names one untried explanation for the output
head reading 22.7 GiB/s in a microbenchmark and 14.25 in the engine: the
microbenchmark sweeps the same 96 MiB three times in a row so its page table
entries stay hot, while the engine sweeps it once with thirty-five layers of
other memory in between.

That is testable, and on this host it is not what is happening. The same 96 MiB
of the mapped checkpoint, four threads, best of four: **32.4 GiB/s swept back to
back and 43.4 with a gigabyte of the rest of the file swept in between** — no
penalty at all, and the difference is this host's own noise. Anonymous memory of
the same size gives 39.4 and 39.8, so a file-backed mapping is not paying for
its four kilobyte pages either.

This does not close the entry, because the entry's own measurements were taken
on the other host and this is not that host. What it does is remove the
hypothesis the entry called "the obvious remaining difference", and the puzzle
reproduces here unchanged — the head phase reads 12.6 GiB/s where a bare sweep
of the same bytes reads 32 to 43 — with the page walk no longer available as the
answer.

### What says it is still the same engine

`app_test.py` against the transformers reference passes every check on all four
prompts. The largest logit gaps are **1.4239, 1.3379, 0.8981 and 1.8306**, which
are 0.8.11's four figures to the last digit printed, and rank one matches on
every prompt. Nothing here was supposed to move them: the integer sum is the
same sum in a different order, which an integer does not care about, and the
float close is the same two multiplies in the same order, sixteen lanes at a
time instead of one row at a time. The correction moved from sixty-four bits to
thirty-two, and `kern_level_wide_ready` is what says it cannot wrap there — a
level is at most 128 in size and a code at most `2^bits - 1`, so the integer sum
is under `128 (2^bits - 1) span`; a zero point is `2^(bits-1)` plus a signed
byte and so at most 256, against a level sum under `128 span`. On the widest
plane this export has, 3072 columns of eight bit codes, the sum of those bounds
is a tenth of `INT32_MAX`.

The unit tests pass on the scalar, SSE2, AVX2 and AVX-512 builds: 727 on the
first three and **736** on the wide one, where nine more assertions run because
the wide block exists there to be checked. The new ones say that it is the one
row path bit for bit at every width and column count the suite carries — floats
compared for equality, not for nearness — that `kern_level_wide_ready` offers it
exactly where a row is a whole number of blocks, and that a host without the
integer dot product is offered it nowhere. `row_count` in that test is seventy,
which is four whole wide blocks and six rows past them, so the block of four
and the single row each take a share of the tail behind it.

---

## 0.8.13 — the fork and the join, off the mutex

### Scope

The route `TODO.md`'s first speed entry named and 0.8.12 measured but did not
take: **make the fork itself cheap.** The pool published a job and collected it
under the mutex, so a fork and a join cost about eight lock acquisitions and
four condition broadcasts — the caller took the lock to publish, each worker
took it again to read what was published, each took it a third time to count
itself done and broadcast, and the caller took it once more to confirm. A
worker that is spinning needs none of that; it is already looking at the
counter.

A decode step forks **277 times**, so this is not `ple feed`'s problem alone —
it is where `ple feed`'s share is largest only because its jobs are the
smallest.

### A note on the host

A third machine, and the numbers are not 0.8.12's either: four cores of a Xeon
at 2.1 GHz, AVX-512 and VNNI, wide build, but a virtualized one where the
shipped export's step floor is **43.44 ms** against the second host's 40.55 and
a decode step reads 760 MiB at 16.6 GiB/s. Ratios carry across hosts and
absolute numbers do not, so every figure below is a ratio taken from eight runs
a side with the two builds alternating, minimum per phase, checkpoint warm in
the page cache.

### What the counters carry now

`task_serial`, `done_count`, `stop_flag` and two new cells — `sleep_count` and
`wait_flag` — are atomics rather than mutex-guarded ints. Three orderings carry
the whole thing, and the third is the one that is easy to get wrong.

**The publish** is a sequentially consistent store on `task_serial` against an
acquire load on the worker's side, so a worker that sees the new serial sees the
`task_call` and `task_state` written before it.

**The completion** is a release increment of `done_count` against the caller's
acquire load, so a caller that sees the count sees the work.

**The handoff between spinning and sleeping** is a store-load pair in both
directions, and it is only safe because both sides are sequentially consistent.
A worker past its spin publishes that it is about to sleep and *then* re-reads
the serial; the caller publishes the serial and *then* reads the sleeper count.
One of the two always sees the other — either the caller finds a sleeper and
takes the lock to wake it, or the worker finds the serial already moved and
never sleeps. With anything weaker than sequential consistency on both stores
and both loads, each side may read the other's stale value and the wake is lost.
The same pair runs on the join, with `wait_flag` for the caller.

So the lock is now what a sleeper is woken *through* rather than what every fork
goes *through*. On the spinning path — the path a decode token is always on — a
fork is one exchange and a few loads, and a completion is one locked add.

### The stress test, which is the point of the version

A lost wake is not a wrong answer once; it is a hang once in a great many
rounds, and a stale read is a wrong answer once in as many. Neither shows up in
a suite that forks twice and checks the bands. `test_platform` now grinds the
pool: 4000 rounds on the spinning path, 2000 on the sleeping one at a width the
host is oversubscribed at, 64 rounds where the caller idles past the spin limit
so every worker is asleep when the next job is published, and 64 where the band
holds long enough that the caller sleeps on the join. Each round stamps its own
number across a 997 element span — prime, so no round divides evenly across the
slices — and leaves a partial beside it, so the caller checks both that every
slot was written *this* round and that what each worker wrote is visible to it.

The two wakes were checked by removing them. With the fork's wake disabled the
suite hangs; with the completion's wake disabled it hangs. Both were confirmed
before the version shipped, because a stress test that passes against a broken
pool is worse than none. The suite also runs clean under ThreadSanitizer, which
is why the acquire is an acquire *load* on each side rather than a relaxed load
behind a fence — the two are equivalent on x86 and only one of them is
something a race detector can follow.

### An empty fork and join

20000 rounds, minimum of five, the engine's own pool:

| threads | before | after | |
| --- | --- | --- | --- |
| 2 | 0.642 us | **0.145** | -77% |
| 3 | 1.312 us | **0.631** | -52% |
| 4 | 3.063 us | **0.603** | -80% |
| 9 (oversubscribed, sleeping) | 142.1 us | 137.5 | -3% |

The last row is the one to read for what did *not* change. An oversubscribed
pool has a spin limit of zero and sleeps on the first look, so it pays the
condition variable exactly as it did; it is here to show that the sleeping path
was not regressed to buy the spinning one.

### The step

A 3 id prompt, four threads, wide build, eight alternating runs a side:

| part | before | after | |
| --- | --- | --- | --- |
| mlp | 20.564 | 20.232 | -1.6% |
| final norm, head | 11.865 | 11.884 | +0.2% |
| q k v | 3.662 | 3.347 | **-8.6%** |
| attn out | 2.835 | 2.739 | -3.4% |
| **ple feed** | 1.533 | **1.310** | **-14.5%** |
| ple lift | 0.985 | 0.953 | -3.2% |
| mlp gate | 0.407 | 0.259 | **-36.4%** |
| step floor | 43.440 | **42.160** | **-2.9%** |
| decode | 23.02 tok/s | **23.72** | +3.0% |

And on a 449 id prompt, where the attention divides across the pool as well:

| part | before | after | |
| --- | --- | --- | --- |
| score, softmax, blend | 4.352 | 4.029 | -7.4% |
| q k v | 3.626 | 3.394 | -6.4% |
| attn out | 2.903 | 2.771 | -4.5% |
| **ple feed** | 1.444 | **1.267** | **-12.3%** |
| mlp gate | 0.411 | 0.248 | **-39.7%** |
| step floor | 47.390 | **46.180** | -2.6% |
| decode | 21.10 tok/s | **21.65** | +2.6% |

The ranking of the movers is the ranking of jobs-per-byte, which is what a
change to the fork rather than to a kernel should produce. `mlp gate` is a
plane that does almost no reading at all and it moves most; `ple feed` is two
384 KiB planes and seventy forks and it moves next; `mlp` is 475 MiB of reading
and moves least. Nothing moved that should not have — `final norm, head`,
`sampler` and `logit cap` are one fork each or none, and all three are flat
inside the noise.

### What this leaves of the entry

0.8.12 put the step's 277 forks at **0.82 ms of 39.5** on the second host. Here
they were 3.06 us apiece, so **0.85 ms of 43.4**, and about 0.68 ms of that is
now gone. The measured step floor moved 1.28 ms, which is more than the fork
arithmetic alone predicts; the difference is that a worker that no longer takes
a lock to read its task also no longer evicts the caller's line to get it, and
that is not something the empty-fork number can see.

`ple feed` is not closed. The plane is 1.31 ms where the four planes that are
not behind run at 50 to 61 G multiply-adds a second, and what is left in it is
the rows rather than the dispatch. The entry's other route — fusing the gate,
the gelu and the lift into one fork, which needs a barrier inside a job that
`pool_group` does not have — is now worth less than it was, because the 35 forks
it would save are 35 times 0.6 us rather than 35 times 3.1.

### Two things deliberately not done

**`pool_seat_list` is still a single static.** It is overwritten by a second
`pool_open` while a first pool is live, which nothing in the engine does and
which the grind test does not provoke either. It is a real bug and it is not
this version's; fixing it here would have put an unmeasured change in the same
diff as the pool rewrite.

**The atomics are builtins behind a shim, not `<stdatomic.h>`.** MSVC's C11
atomics are recent and gated, and the engine builds under `cl`; five operations
behind `_Interlocked*` on that compiler and `__atomic_*` on the others is
smaller than the configuration test the header would need.

---

## 0.8.14 — one bit matrix where the shift and the mask were two, and what the output head has actually been doing

### Scope

Two things, and the second is worth more than the first even though it is not a
change to the engine at all.

The first is a kernel: the packed field a code plane's inner loop takes out of
a byte is a per-byte linear map over GF(2), so `vgf2p8affineqb` does it in one
instruction where a variable shift and a mask did it in two.

The second is a diagnosis. Making that change and watching which planes moved
answered `TODO.md`'s second speed entry — the output head reading a third of a
bare sweep with neither the memory nor the instruction count able to explain
it — and the answer is that **the head has never been running the integer
kernel.**

### The host

The third machine again — four cores of a Xeon at 2.1 GHz, virtualized, step
floor 43.44 ms before this version. Eight alternating runs a side, minimum per
phase, checkpoint warm. Output is bit-identical either way: `logits` and a
48 token greedy `chat` compare byte for byte.

### The affine take

`KERN_LEVEL_TAKE_2` broadcast sixteen packed bytes, shifted each quarter of the
broadcast down by its own bit position and masked two bits out of every byte.
The shift is by 0, 2, 4 or 6 places and the mask keeps the low two bits, so
although `_mm512_srlv_epi32` shifts a whole dword, **the low two bits of output
byte `b` come from bits `[s, s+2)` of input byte `b` and from nowhere else.**
Every field is taken from within its own byte, which is precisely the shape
`vgf2p8affineqb` is for: it gives each byte of the result as an eight by eight
bit matrix multiplied by that byte of the source, with the matrix a per-qword
operand so the four quarters can each have their own.

Output bit `k` is the parity of `matrix.byte[7 - k]` against the source byte, so
taking source bit `s + k` into result bit `k` is the matrix whose byte `7 - k`
is `1 << (s + k)` and whose other bytes are zero. Bits above the field's width
have no row and come out zero, which is the mask, for free. `kern_level_matrix`
writes that; `KERN_LEVEL_PLAN` lays four of them out to match the quarters the
old `step_wide` addressed, and the same construction serves four bits with two
rows apiece.

The two bit inner loop is now `vbroadcasti32x4`, `vgf2p8affineqb`, `vpdpbusd` —
three instructions per sixty-four codes against four. Eight bit codes need no
unpack and are untouched.

`gfni` is asked of the host the way `vnni` already was, because it does not
travel with it: Cascade Lake has `vnni` and no `gfni`, Ice Lake has both. Where
the compiler says `-march=native` would not define `__GFNI__`, the shift and the
mask are still there.

### What moved, and what did not

| part | before | after | |
| --- | --- | --- | --- |
| mlp (4 bit) | 21.137 | **19.021** | **-10.0%** |
| q k v (4 bit) | 3.695 | **3.015** | **-18.4%** |
| attn out (4 bit) | 2.875 | **2.526** | **-12.1%** |
| ple feed (8 bit) | 1.339 | 1.302 | -2.8% |
| **final norm, head (2 bit)** | 12.073 | 12.356 | **+2.3%** |
| step floor | 47.550 | **44.740** | **-5.9%** |
| decode | 21.03 tok/s | **22.35** | **+6.3%** |
| prefill | 84.00 tok/s | 86.12 | +2.5% |

On a 3 id prompt the same eight runs give the step floor 43.110 to 41.340 and
decode 23.20 to 24.19.

Two of those rows are the result and the third is the finding. The four bit
planes are a quarter of their inner loop lighter and move by ten to eighteen
percent, which settles that they were issue-bound rather than memory-bound —
they read 475 MiB a step at 22 GiB/s where a bare four thread sweep on this host
reaches 32 to 43, and now they read the same bytes faster without reading fewer
of them. `ple feed` is eight bit, has no unpack, and correctly does not move.

**And the head, which is two bit and has the unpack, does not move either.**

### The output head, answered

`TODO.md`'s second speed entry has been open across three versions. 0.8.11
removed the row epilogue's calls and the head did not move. 0.8.12 folded
sixteen rows into one close and the head was the one plane it did not move.
0.8.12 also killed the page walk hypothesis outright. The entry's own summary
was that the head is bound by neither of the two things it could be bound by,
and that the next step was to count retired instructions rather than estimate
them.

This version removed an instruction from the head's supposed inner loop — a
quarter of it — and the head did not move. That is the counting experiment in
the only form this host allows, and it says the loop under test is not the loop
being run.

It is not. Instrumenting `kern_mat_vec_band` by plane shows the 262144 row
plane taking the **float fallback**, `kern_row_code`, on every decode step —
1610612736 column-products over four steps, which is 4 × 262144 × 1536 exactly.
The integer path never sees it.

The reason is one number in the export. `kern_level_ready` requires
`sheet->enter_gain > 0`, the plane's `input_activation_scale`, because the
integer path works by rounding the activation onto that step's grid and
requiring it to land exactly. Every other projection has one — `q_proj` 0.0728,
the audio tower's first 0.1591, and so on. **`lm_head.input_activation_scale` is
0.0.** The export ships the head with no calibrated input step, so the head's
input is not quantized, so the plane can never be on a grid, so it takes the
float path, every token, by construction.

Everything the entry could not explain follows from that:

- **The GiB/s gap.** The head reads 97 MiB at 7.8 GiB/s where the four bit
  planes read at 20 to 22. That is not two bit against four bit; it is
  `kern_dot_code`'s float spread against `vpdpbusd`. In multiply-adds the head
  gives 33 G a second against the mlp's 52 on this host.
- **The microbenchmark that ran faster than the engine.** The entry's 22.7
  GiB/s was "the same kernel on the same shape" — but it was the integer
  kernel, and the engine runs the float one. The two were never the same
  measurement.
- **Every fix that did not move it.** 0.8.11's epilogue, 0.8.12's fold and this
  version's affine take are all in the integer path. The head is not in the
  integer path.

The entry is closed as a question. What is left is a different and much better
posed one, which `TODO.md` now carries: the head runs `kern_dot_code`'s two bit
loop at five instructions per sixteen codes — a broadcast, a variable shift, a
mask, a convert and an fma — where the integer path spends four per sixty-four,
and that loop has never had a version written for it.

### Why the head was not fixed here as well

Three routes were considered and none of them belongs in this version.

**Quantizing the head's input to a step the engine picks.** That is what would
put it on the integer path, and it is not exact: the export declined to
calibrate this activation, and choosing a step here would change the logits for
a speed win, which is a decision about output quality rather than a kernel
change. It wants its own version and its own measurement of what it costs.

**A cheaper float loop.** Real and available — a broadcast amortized over
sixty-four codes rather than sixteen takes the loop from twenty instructions to
seventeen — but it needs the activations staged in the unpack's order, the way
`kern_level_stage` already stages levels for the integer path, and that is a
new staging pass and a new correctness surface. Also its own version.

**Not scoring 262144 rows at all.** `TODO.md`'s fourth speed entry, unchanged by
any of this except that the plane it would prune is now known to be five times
more expensive per byte than the entry assumed.

### A note for whoever reads the old entry

Do not re-derive the head's cost from GiB/s, and do not re-derive it from the
integer path's instruction count either. Both were done, both were careful, and
both were measuring a kernel the head does not run. The first question to ask of
any plane that looks anomalous is which of the two paths it is on, and
`kern_mat_vec_band` is four lines of instrumentation away from saying so.

---

## 0.8.15 — the mask and the widening as one lookup, in the loop the head actually runs

### Scope

0.8.14 found that the output head runs `kern_dot_code`'s float spread rather
than the integer path, and left three routes. This is the first and smallest of
them: the two bit float loop, which had never had a version written for it, and
which is 27% of a decode step on this host.

### The host, and what it can actually do

Third machine, four cores of a Xeon at 2.1 GHz, virtualized, AVX-512 with VNNI
and GFNI. Eight alternating runs a side on a 3 id prompt and six on a 449 id
one, minimum per phase, checkpoint warm.

Worth writing down because every entry above reasons from a host's sweep and
this host's had not been measured: **a bare four thread sweep of the mapped
checkpoint reaches 49.80 GiB/s here**, 24.60 at two threads and 13.04 at one —
linear, so the memory is not the bound at any width. A decode step reads 760 MiB,
which at that rate would be 14.9 ms against the 43.79 ms the step floor actually
is. Nothing in this engine on this host is memory-bound. That is the context for
every phase that moved today.

### Four instructions a vector rather than five

The loop took a dword of packed codes, broadcast it across all sixteen lanes,
shifted each lane down by its own code's bit position, masked two bits, widened
to float and multiplied into the accumulator: `vpbroadcastd`, `vpsrlvd`,
`vpandd`, `vcvtdq2ps`, `vfmadd132ps`.

The mask and the widening are one instruction. After the shift, a lane holds its
own code in bits zero and one — and the *next* code in bits two and three, since
the shift moved the whole dword. So the low four bits of the lane are a number
from zero to fifteen whose remainder on four is the code that lane wants. That
is exactly the field `vpermps` indexes with. A sixteen entry table of
`0, 1, 2, 3` repeated four times therefore returns the code already a float,
with the mask implied by the table repeating and the conversion implied by the
table's contents.

The lanes, the codes, the two accumulators and the order they are added in are
all what they were, so this is the same sum to the last bit. `logits` on the
shipped export compares byte for byte against the previous build.

### The step

| part | 3 id prompt | | 449 id prompt | |
| --- | --- | --- | --- | --- |
| **final norm, head** | 12.232 to **11.216** | **-8.3%** | 12.185 to **11.172** | **-8.3%** |
| step floor | 41.580 to **40.600** | -2.4% | 44.740 to **43.790** | -2.1% |
| decode | 24.05 to **24.63** tok/s | +2.4% | 22.35 to **22.84** | +2.2% |

Every other phase is inside the noise on both prompts, which is what a change to
one loop that one plane reaches should look like. The head is the only plane in
the step on the float path, and it is the only plane that moved.

### What is left in the head, and why it is not here

The loop is now four instructions per sixteen codes: a broadcast, a shift, the
lookup, the multiply-add. Sixteen per sixty-four codes against the integer
path's four.

Three of those sixteen are broadcasts that could be one. `vbroadcasti32x4`
takes sixteen bytes — sixty-four codes — and four shifts of it reach every code
in them, which is thirteen instructions per sixty-four rather than sixteen. It
is not done here because the four quarters come out in the unpack's order rather
than the column's, so the activations would have to be laid down in that order
the way `kern_level_stage` already lays down levels for the integer path. That
is a new staging pass and a new correctness surface, and it belongs in a version
that is about it rather than riding on a four line change that is provably the
same arithmetic.

Against the ceiling: the head reads 97 MiB, which at this host's 49.80 GiB/s is
1.9 ms. It is 11.17. The loop at four instructions a vector, on the ports this
machine has, is worth about 4.5 ms of that by arithmetic, and the gap between 4.5
and 11.17 is not yet accounted for. `TODO.md` carries it with the other two
routes.

---

## 0.8.16 — the output head was waiting on its own accumulator

### Scope

The rest of `TODO.md`'s head entry, and it was not the instruction count after
all. 0.8.15 took the two bit float loop from five instructions a vector to four
and got 8.3%; the arithmetic said the loop should then be worth about 4.5 ms and
it was 11.17. This is the missing factor, and it is latency.

### The chain

A row of the output head is 1536 columns, which is 96 vectors, and
`kern_dot_code` carries **two** accumulators. So each of them is a chain of
forty-eight dependent `vfmadd132ps`, and a multiply-add is four cycles deep
against two a cycle of throughput. The row's floor is the depth of its chain —
about 192 cycles — where its ports would allow about 48. **Four times, and it is
not the loop body at all.**

More accumulators fix it and change which slot is added to which, so the sum
moves. Four rows at a time fix it without touching the sum: each row keeps its
own pair of accumulators, its own slots and its own order, and the eight chains
cover each other. The activation vector is loaded once for the four, which is
the same trade the integer path's row block already makes.

This is exactly what 0.8.12 did to the integer path — and the head was the one
plane 0.8.12 could not reach, because the head is not on that path. It took
0.8.14 to find that out and 0.8.15 to make the float loop worth blocking.

### The step

Eight alternating runs a side on a 3 id prompt, six on a 449 id one, minimum per
phase, checkpoint warm:

| part | 3 id prompt | | 449 id prompt | |
| --- | --- | --- | --- | --- |
| **final norm, head** | 11.203 to **5.994** | **-46.5%** | 11.298 to **5.896** | **-47.8%** |
| step floor | 41.100 to **34.950** | **-15.0%** | 43.880 to **38.310** | **-12.7%** |
| decode | 24.33 to **28.61** tok/s | **+17.6%** | 22.79 to **26.10** | **+14.5%** |

Nothing else moves outside the noise: the block is reached by one plane. The
head is 97 MiB at **16.2 GiB/s** where it was 8.7, and 67 G multiply-adds a
second where it was 33.

`logits` and a 48 token greedy `chat` on the shipped export are byte for byte
what the build at the start of this session produced, across all four of
0.8.13 to 0.8.16.

### Four rows, and eight measured against them

Eight was built and run, and it is worse on the plane it is for: the head 5.989
ms against 6.240, six alternating rounds a side, with the step floor inside the
noise either way. Sixteen live accumulators plus the two activation vectors and
the constants is most of the register file, and four chains a row already cover
a four cycle multiply-add. That agrees with 0.8.8 and 0.8.11, which found the
same wash at eight for the same reason on the other path.

### Where the head stands now

5.99 ms of a 34.95 ms step, 17.1% against 27.3% before this version. A bare four
thread sweep of the same 97 MiB on this host is 1.9 ms, so the plane is now
about three times its memory floor rather than six.

What is left in it is the loop, and `TODO.md` carries it: three of the sixteen
instructions per sixty-four codes are broadcasts that one `vbroadcasti32x4`
could replace, at the price of staging the activations in the unpack's order.
That is now a change to a loop that is no longer latency-bound, so for the first
time the instruction count is the thing to count.

### The four versions together

From the build at the start of this session to this one, on a 3 id prompt, four
threads, wide build:

| | before | after | |
| --- | --- | --- | --- |
| decode | 23.02 tok/s | **28.61** | **+24.3%** |
| step floor | 43.440 ms | **34.950** | **-19.5%** |
| final norm, head | 11.865 | **5.994** | -49.5% |
| mlp | 20.564 | 19.312 | -6.1% |
| q k v | 3.662 | 3.069 | -16.2% |
| attn out | 2.835 | 2.524 | -11.0% |
| ple feed | 1.533 | 1.281 | -16.4% |
| mlp gate | 0.407 | 0.251 | -38.3% |

Same weights, same answer to the byte.

---

## 0.9.0 — the verify side of speculative decoding, and what it is worth

### Scope

The three pieces of `TODO.md`'s speculative decoding entry that can be built and
measured without a proposer, and which together decide whether the rest of it is
worth building:

1. **Every position's logits out of one pass**, which the engine did not have.
2. **A cache that can be put back**, which is the entry's named correctness trap.
3. **A measurement that brackets the payoff**, using two proposers that could
   never be written for real work — one always right, one always wrong.

No proposer is written here, and none should be until the number in the last
section is looked at.

### All-position logits

`session_pass` computed the head for the last lane only, because a decode step
wants one distribution and a prefill wants none until its final chunk.
`PASS_LOGIT_ALL` runs it for every lane instead — normalising each lane into its
own row of `scrap_room` and handing the whole block to `session_lift_many`, so
the head is one batched product rather than a loop of single ones. That matters
more than it looks: the batched code path decodes a row of the head once and
reads it with every lane, so a block of eight sweeps the head's 96 MiB once and
not eight times.

`logit_room` grows from one row to one a lane the first time a caller asks. It
is not grown at `session_open` because a row of this vocabulary is a megabyte
and a session that never guesses should not carry sixteen of them.

Ordinary decode is untouched: `logits` on the shipped export is byte for byte
what it was.

### The cache that can be put back

`TODO.md` called this the correctness trap and it was right about why. A plain
transformer appends its cache, so undoing a block is a counter. This one does
not: **twenty-eight of thirty-five layers are sliding-window layers holding a
ring of `slide_span` rows**, and a block written past the ring's length has
overwritten rows an earlier position still needs. Seven layers are full
attention and would be happy with the counter. Some layers read another layer's
cache and own none, and unwinding those twice would corrupt the layer that does.

What makes it tractable is a bound the entry did not name: **a block is at most
`KERN_LANE_LIMIT` rows and a ring is at least `slide_span`, so a block never
laps itself** and the rows it will overwrite are known before it runs. Saving
them is sixteen rows a side a layer — a fixed scratch — and undoing is the same
copy back. `session_guess_limit` reports the bound and `session_guess` refuses a
block that would exceed it, which is what a checkpoint with a four row window
needs and what this export never reaches.

The API is three calls. `session_guess` runs a block and hands back a row of the
vocabulary a lane; nothing else may touch the session until `session_guess_keep`
says how many lanes to keep; zero puts the session back exactly where it was.

The peaks are the one thing not restored on a partial keep. They are a
high-water mark rather than an input to any sum, so a kept block may leave one
carrying a rejected lane's magnitude — an overstatement of what the cache was
asked to hold, which is the safe direction for a diagnostic about whether a
calibrated range suffices. A full undo restores them, because a full undo has to
leave nothing behind.

### The test, and the bug it was checked against

`test_guess` runs on the **synthetic** checkpoint and not the shipped one, for a
specific reason: its sliding window is **four**. A block of four therefore
overwrites every row of the ring, which is the case the undo exists for and the
case a 512 row ring would not reach in a test of a few dozen tokens.

The assertion is a hash of everything a block could have moved — every layer's
key and value store, the peaks, both counters and the id history — taken before
the block and after it is dropped. Byte for byte, not to a tolerance: a
tolerance here would pass a cache that still held a rejected guess.

It was checked by writing the bug the entry warns about. Replacing the row
restore with nothing — trusting `fill_count`, as a plain transformer could —
makes the suite fail on exactly that assertion. The test also covers the refusal
of a block longer than the ring, that a step and a prime are refused mid-block,
and that the same block run twice across a drop gives the same logits to the
bit.

### What a block is worth, bracketed

`igllm guess` runs the same greedy continuation three ways: plain, with a
proposer that always guesses right, and with one that always guesses wrong. The
first is the ceiling of any proposer and the second is its floor. All three are
held to the plain run's token stream, so the task checks the block path as much
as it measures it.

On the shipped export, four threads, 32 tokens, two prompts:

| block | proposer | tok/s | ms a round | committed a round | vs plain |
| --- | --- | --- | --- | --- | --- |
| 2 | oracle | 27.6 | 72.5 | 2 | 1.06x |
| 4 | oracle | 39.3 | 101.8 | 4 | **1.51x** |
| 8 | oracle | 56.2 | 142.4 | 8 | **2.15x** |
| 16 | oracle | 57.8 | 277.2 | 16 | 2.21x |
| 4 | null | 9.8 | 102.4 | 1 | 0.38x |
| 8 | null | 7.7 | 129.2 | 1 | 0.30x |

Every row printed `matches plain`.

**Two numbers come out of this and they are the whole answer.**

**The ceiling is 2.2x.** A proposer that is never wrong doubles the engine and
does not treble it, and past a block of eight it stops improving. That is far
short of what the idea is worth on a memory-bound engine, and the reason is
identifiable rather than mysterious.

**Break-even needs about 40% of guesses accepted.** A round of eight costs 142.4
ms against a plain step's 38.3, so it has to commit 3.7 tokens to pay, which is
2.7 accepted of 7. At a block of four the bar is worse, 1.7 of 3.

### Why the ceiling is 2.2 and not 6

A block shares the weight *sweep* across its lanes and does not share the
*arithmetic*. On an engine that is memory-bound the first is nearly all of the
cost and an extra lane is nearly free; this one is not memory-bound — 0.8.15
measured a bare four thread sweep at 49.80 GiB/s against a step floor that reads
760 MiB in 35 ms — so the arithmetic is most of what a lane costs.

Measured from the rounds above, **an extra lane costs about 13 ms against a
plain step's 38**, so the asymptotic ceiling is about 3x and a block of eight
reaches 2.2 of it.

**Six of those 13 ms are the output head**, and that is the part worth naming.
Verifying a position means asking what the model would have produced there,
which means the full 262144 row head for every lane. `kern_row_code_many` shares
the head's 96 MiB across the lanes correctly — it decodes a row once and every
lane reads it — but this engine is arithmetic-bound and the 402 M multiply-adds
a lane are not shared and cannot be.

So the head is not merely one of the costs of speculative decoding here; it is
half of the marginal lane. `TODO.md`'s "stop scoring 262144 rows to pick one"
and this entry are the same piece of work seen from two sides, and the ordering
below is written on that.

### What this does not settle

Whether any real proposer reaches 40% on this model. That needs the proposer,
and `TODO.md` now says which one to write and what to hold it to. What is
settled is that the verify side works, that it costs 13 ms a lane, and that the
best it can ever be worth in this engine's present shape is 2.2x.

---

## 0.9.1 — the batched output head, which had neither of the two things the one lane head grew

### Scope

`TODO.md`'s speculative decoding entry names this as the first thing to take
after 0.9.0, and says why: **six of the thirteen milliseconds an extra
speculative lane costs are the output head**, because verifying a position means
asking what the model would have produced there, and that is the full 262144
rows for every lane. 0.8.15 and 0.8.16 made the one lane head three times
cheaper and neither of them reached the batched one.

Nothing else is changed. Decode is one lane and is untouched; `logits` on the
shipped export is byte for byte what it was, and a greedy `chat` and `complete`
produce the same text.

### What the batched head was doing

The many lane float path is `kern_row_code_many` over two functions:
`kern_code_spread`, which writes a group of a row out into a scratch buffer as
floats, and `kern_dot_real_many`, which reads that scratch back four lanes at a
time. Neither could ever have had what the one lane path was given.

0.8.15 replaced the two bit mask and widening with one `vpermps` against a
repeating table — the cheapest unpack in this file, and it exists only inside
`kern_dot_code`'s loop. 0.8.16 found that a float row of the head is bound by
the depth of its own accumulator chain rather than by its ports, and covered it
by taking four rows at once so eight chains cover each other; that is inside
`kern_row_code_rows`. A spread's scratch has no unpack to make cheaper — it has
already been paid, at the width the spread was written for — and it turns every
multiply-add on the other side into a load, because the codes have to be read
back.

Counted against the ports, on 32 columns and 4 lanes — 128 multiply-adds:

| | loads | arithmetic |
| --- | --- | --- |
| spread, then `kern_dot_real_many` at 256 bits | 4 of the row, 16 of the lanes | 16 multiply-adds, plus the spread's own unpack and its stores |
| `kern_dot_code_many` at 512 bits | 2 dwords, 8 of the lanes | 2 broadcasts, 2 shifts, 2 lookups, 8 multiply-adds |

### What it does now

`kern_dot_code_many` is the batch's answer to what `kern_row_code_rows` is for
one lane. The group is decoded where the one lane path decodes it — a broadcast,
a variable shift and the `vpermps` lookup, three instructions for sixteen codes
— and multiplied straight into each lane's pair of accumulators, four lanes at a
time. Eight chains, which is the count 0.8.16 found sufficient; the decode is
spent once for the four lanes rather than once for each; and the scratch is gone
from both sides of it.

Two bits and AVX-512 only, guarded by `kern_code_rows_ready`, which is the
predicate the one lane block already uses and is the same question. That is the
width and the host the one plane on this path has. Lanes past the last block of
four go through `kern_dot_code` itself, which is the one lane path's own loop.
Every other width and every other backend takes the spread exactly as before.

### The sum, and what holds it

This agrees with `kern_row_code` to a float's tolerance and not to its last bit,
because a 512 bit accumulator adds a lane's slots in a different order than a
256 bit one. That is what a batch has always done here — `back_mat_mat matches a
lane at a time` is the test that says so, at a tolerance rather than an equality,
and the batched path has never been bit-identical to a sequence of single steps
for the reason 0.9.0 wrote down. What is held exactly is the thing a caller
sees: `igllm guess` prints `matches plain` on every block and every proposer,
which is the block path's token stream against the plain one, end to end.

### What it is worth

Six runs alternating between the two builds, `igllm guess --serve 16` on the
shipped export at four threads, the minimum of each. The host is a four core
Xeon at 2.8 GHz with AVX-512 and VNNI, and it is not a quiet one: `plain` itself
ranged 34.8 to 42.1 ms a token across the twelve runs, so the rows below are the
minima and the spread is quoted rather than hidden.

| block | proposer | rounds | before | after | |
| --- | --- | --- | --- | --- | --- |
| 2 | null | 16 | 64.86 | 57.90 | −10.7% |
| 4 | null | 16 | 75.79 | 66.55 | −12.2% |
| 8 | null | 16 | 124.36 | 108.69 | −12.6% |
| 16 | null | 16 | 178.87 | 164.48 | −8.0% |
| 4 | oracle | 4 | 82.15 | 72.32 | −12.0% |
| 8 | oracle | 2 | 142.59 | 131.74 | −7.6% |

The null rows are the ones to read. A null proposer commits one token a round,
so `--serve 16` is sixteen rounds of it and the figure is a mean over them; an
oracle at a block of sixteen is **one** round and is a single sample with a page
fault in it, which is why that row is not quoted at all and why its numbers
swung by a third between runs of the same binary.

At a block of eight the round is **124.36 ms to 108.69**, and against a plain
step of 37.26 that is a marginal lane of 12.44 ms falling to 10.56 — most of the
way through the head's half of it, which is what the entry predicted.

Prefill does not move, and that is expected rather than disappointing: a prefill
pays the head once for its whole chunk. Eight alternating runs on a 228 id
prompt, best of each: 78.23 tokens a second before and 80.76 after, which is
noise in both directions.

### What is left in it

The batched head is now on the same unpack and the same chain depth as the one
lane head, so the two routes still open in `TODO.md` are the two that were
already there: the broadcast the loop still spends three of sixteen instructions
on, and not scoring 262144 rows at all. The second of those is the larger, and
0.9.0 gave it a second customer — greedy verification asks whether the proposed
id is the argmax, which is a bound and not a distribution.

---

## 0.9.2 — the mlp's chain, counted and refused, and the sweep that says why

### Scope

No code changes. `TODO.md`'s first entry — the feed-forward planes, half of
every token, with no history of attempts on it — names one thing to check before
anything cleverer, and this is that check, on the reference host, with the
numbers written down so it is not checked a third time.

> Start where 0.8.16 started: **count the loop against the ports, and count the
> chain against the loop.** `kern_row_code_level_rows` carries one accumulator a
> row over four rows. A 1536 column row at four bits is twenty-four blocks, so
> each chain is twenty-four dependent `vpdpbusd` — five cycles deep on this
> class of host against two a cycle of throughput, and four chains to cover it.

The arithmetic is right and the conclusion does not follow. Two accumulators a
row were built, on both integer loops, and both are refused.

### The one lane row block: a wash

`KERN_LEVEL_ROWS_LOOP` was given a paired loop — two blocks an iteration, two
accumulators a row, eight chains where there were four, the level vector still
loaded once for the four rows, and the second accumulator folded into the first
inside the macro so no caller could tell. It is bit-identical by construction:
a `vpdpbusd` accumulator is an `int32` sum of products, integer addition is
associative, and `logits` on the shipped export came back byte for byte
unchanged, which is the check that says the build did what it says.

Three runs each, alternating builds, `bench --serve 96 --verbose`, the minimum
of the `mlp` phase:

| | mlp, ms a step |
| --- | --- |
| four chains | 18.989, 19.745, 21.480 |
| eight chains | 18.959, 20.648, 21.105 |

18.989 against 18.959. There is nothing there, in either direction.

### Why not, and the number the entry was missing

The entry's ratio — "two point one times its own memory floor" — is the third
host's, where a bare four thread sweep reaches 49.80 GiB/s. **It is not this
host's.** A bare read-only sweep of a 2.4 GiB buffer here, best of four:

| threads | 1 | 2 | 4 |
| --- | --- | --- | --- |
| sequential read | 10.06 GiB/s | 17.09 | **31.29** |

which is the 32.18 of 0.8.9's table on the same machine, a year of neighbours
later. The mlp reads **475.3 MiB at 24.44 GiB/s**, so on this host it is at
**78% of a bare sweep**, and its memory floor is 14.83 ms against the 18.96 it
takes: **1.28 times its floor, not 2.1.** Even a plane with no arithmetic at all
in it would save 4.1 ms of a 36.9 ms step here — 11% — and 0.8.9 measured the
four bit code path itself at 29.47 GiB/s in isolation, so the honest ceiling on
this entry is nearer 17% of the plane than to anything larger.

A loop that is within a fifth of what its memory will hand over cannot be
latency-bound, whatever counting its chains says. The chains are real; there is
simply no port pressure behind them to relieve. On the third host, which sweeps
at 49.80 GiB/s and where the same plane sits at 45% of a sweep rather than 78%,
the answer could well be different — and that is where this should be retried,
not here.

### The batched loop: 15% worse, and a better hypothesis than the first

`KERN_LEVEL_MANY_LOOP` looked like the case the argument actually fits. A batch
reads the plane once and multiplies it by every lane, so the arithmetic a byte
carries is the lane count and memory stops binding after the first lane or two —
which is exactly where a latency-bound loop shows, and it is the loop a
speculative block spends its marginal lane in. The same pairing was built there:
two accumulators a lane, two blocks an iteration, both blocks still decoded once
for the four lanes.

Three runs each on a 228 id prompt, best of each: **prefill 78.28 tokens a
second against 66.22** — the deeper loop is 15% *worse*. Eight accumulators, two
decoded code vectors, four lane pointers and the plan's three constants do not
fit, and what the chains gain the spills lose several times over.

So the batched integer path is at its register limit and not at its latency
limit, and that is worth knowing before anything else is written into it.

### What this leaves

The mlp entry stays open and its hypothesis does not. What is ruled out on it is
now: the bit width (0.8.11), eight rows a block (0.8.11, 0.8.16), software
prefetch (0.8.11), the page walk (0.8.12), and the accumulator chain at both
widths (here). What is left is a plane at 78% of the host's memory, which is a
smaller prize than the entry was written for, and the entry now says so.

---

## 0.9.3 — a proposer, and the flag that puts a block behind an ordinary turn

### Scope

Steps 3 and 5 of `TODO.md`'s speculative decoding entry. 0.9.0 built the verify
side and measured what a block is worth; 0.9.1 halved the head's share of a
lane. What was missing was something to propose the block and a way for a caller
to ask for one.

- **`app_scout`**, an n-gram proposer with no second model in it.
- **an `n-gram` row in `igllm guess`**, beside the oracle and the null, so the
  thing that ships is measured against the bracket rather than against itself.
- **`--guess <lanes>`** on `chat` and `complete`, greedy only, with a line
  after the run saying what it bought.

Step 4 — sampling — is deliberately not here, and the flag says so rather than
sampling from a distribution the block would have skewed.

### The proposer

`scout_draw` asks one question: **what did this token stream do the last time it
was here?** Take the last few ids, find the most recent earlier place the same
ids appeared, and propose what followed them there. The longest reach that
matches anywhere wins, and among the places a reach matches, the latest wins.

There is no model, no training and no second set of weights — the whole of it is
a growing array of ids and a backward scan. A scan of the longest window this
export has is a few hundred thousand `int32_t` comparisons, under a fifth of a
millisecond against a round of a hundred, so the obvious loop is the right loop
and an index would be complexity for nothing.

A scout is told the prompt and then **only the tokens the model has agreed to**.
One told about its own guesses would learn from them.

### The shortest reach is two, and that is the whole of why the flag is cheap

The obvious floor is one — match on a single id — and it is wrong on both
workloads at once. A single id matches somewhere in almost any stream, so a
reach of one is not a memory of anything: it draws nearly every round and is
right almost never. At `--guess 4` on the shipped export, a prompt whose answer
quotes it against free generation:

| shortest reach | quoting | free |
| --- | --- | --- |
| 1 | 46.18 tok/s, 90% of 30 kept | 20.37 tok/s, 0% of 53 kept |
| **2** | **47.28**, 100% of 27 | **24.80**, and it draws nothing at all |
| 3 | 43.98, 96% of 27 | 25.33, and it draws nothing at all |

Two is better than one on the workload the scout is for *and* on the one it is
not, which is the rare shape of an argument that needs no trade-off: dropping
the reach of one throws away guesses that were wrong anyway. Three costs the
quoting case a fifteenth for very little on the other side.

The second half of the same saving is in the caller. Where the scout proposes
nothing the round is an ordinary `session_step` rather than a block of one lane,
which is cheaper by the block path's bookkeeping. Together these two are what
takes free generation from 0.74 of the plain rate to 0.95.

### What it is worth

`chat --heat 0 --serve 64` on the shipped export at four threads, best of three
alternating runs each, and the emitted text is byte for byte the plain text in
every case:

| prompt | plain | `--guess 4` | |
| --- | --- | --- | --- |
| an answer that quotes the prompt | 26.02 tok/s | **47.73** | **1.83x** |
| free generation | 27.63 tok/s | 26.32 | 0.95x |

The tally line says why, and it is the two numbers that decide any proposer:

```
guess   3.08 tokens a round over 12 rounds, 100% of 27 guesses kept
guess   1.00 tokens a round over 64 rounds, 0% of 0 guesses kept
```

### Against the bracket

`TODO.md` asked for the scout to be held to **committed tokens per millisecond**
against `igllm guess`'s oracle and null rows rather than to an acceptance rate,
and on the two workloads that differ. It now runs as a third row of that table,
with the same plain-step fallback the flag has, so what is measured is what
ships. On the shipped export at four threads:

Free generation, `guess --serve 32` on a three id prompt:

```
block  proposer    tok/s  ms a round  committed of drawn  vs plain  stream
4      oracle      49.68       80.52       4.00     100%     1.93x  matches plain
4      n-gram      24.41       40.97       1.00       0%     0.95x  matches plain
4      null        13.08       76.47       1.00       0%     0.51x  matches plain
8      oracle      59.40      134.69       8.00     100%     2.31x  matches plain
8      n-gram      23.55       42.45       1.00       0%     0.92x  matches plain
8      null         7.98      125.39       1.00       0%     0.31x  matches plain
```

An answer that quotes its prompt, `guess --serve 48`:

```
block  proposer    tok/s  ms a round  committed of drawn  vs plain  stream
4      oracle      49.73       80.44       4.00     100%     2.02x  matches plain
4      n-gram      41.78       63.83       2.67      94%     1.69x  matches plain
4      null        12.74       78.52       1.00       0%     0.52x  matches plain
8      oracle      52.08      153.61       8.00     100%     2.11x  matches plain
8      n-gram      44.27       83.41       3.69      95%     1.80x  matches plain
8      null         7.37      135.66       1.00       0%     0.30x  matches plain
```

**On the workload it is for, the scout reaches 1.80x against a ceiling of
2.11x** — 85% of everything a proposer that is never wrong could give — at 95%
of guesses accepted and 3.69 tokens a round. On the workload it is not for it
draws nothing at all and costs the block path's accounting. Every row matches
the plain stream, which is the check that the block path verified correctly and
the undo left nothing behind.

**So this is a flag and not a default, and `TODO.md` said to decide that on this
measurement.** On text that quotes its input — summarising, editing, answering
about a document, repairing code that is in the prompt — the continuation of a
phrase is usually in the prompt already and the scout is right nearly every
time. On free generation it has only what it has written itself, draws nothing,
and costs a twentieth for the block path's accounting.

### What the decode rate now counts

A block was invisible to the tally: `session_step` counted tokens, seconds and
bytes, and `session_guess` counted none of them, so a run under `--guess`
reported `decode 0.00 tok/s`. It now counts into the same three, and the way it
counts is the point of the whole feature: the **weights once**, because every
lane of a batched product reads the same row, and the **cache once a lane**,
because each lane attends over its own prefix. `session_guess_keep` adds the
tokens actually committed. On the quoting run that is `reads 227.0 MiB a token`
against a plain step's 766.3 — the same tokens, a third of the memory.

The phase division stays off for a block, for the reason a prime pass is kept
out of it: it is armed for a decode step, and a block runs the same graph
several lanes wide.

### Greedy only, said out loud

`--guess` with a temperature is refused rather than silently ignored.
Verification here is an argmax comparison, which is the right rule at zero and
the wrong one above it — speculative decoding under a temperature needs the
modified rejection rule, accept with probability `min(1, p/q)` and resample from
the difference, and an n-gram proposer has no `q` to divide by. That is step 4
of the entry and it needs a proposer that carries a distribution, or a decision
that the feature is greedy-only forever.

Two other honest limits. `--guess` is on the single-turn path — `chat`,
`complete` — and not yet inside `chat --loop`, where a scout would want to
carry across turns; that is where prompt lookup would be at its very best,
because a follow-up question about the same document quotes both the document
and the previous answer. And a prompt whose last id is a soft token from a
tower takes the plain loop whatever the flag says, because a block cannot carry
an embedding row.

---

## 0.9.4 — the proposer carried across turns, which is where prompt lookup belongs

### Scope

0.9.3 shipped `--guess` on the single turn tasks and said what was left: the
scout was per-run, so `chat --loop` could not have it. That is the case prompt
lookup is *best* at and the one it was missing — a follow-up question about the
same document quotes both the document and the answer before it.

`app_scout` now belongs to the conversation rather than to the turn. `main_talk`
carries one, opened the first time `--guess` needs it, told each turn's ids as
they arrive and every committed token after that.

### What it is worth

Two turns: one that plants a passage, then one that asks for it back. `--serve
48`, greedy, `--guess 4`, best of three alternating runs, whole process wall
clock including the load and both prefills:

| | plain | `--guess 4` | |
| --- | --- | --- | --- |
| a passage planted, then quoted back | 5.56 s | **4.33 s** | **1.28x** |

The tally says where it came from. The first turn is mostly the model's own
words with the passage quoted inside them, and the scout is already useful
there — 2.09 tokens a round at 83% of guesses kept. The second turn is almost
entirely quotation, and it reaches 3.08 tokens a round at 82%. A scout that
started fresh at each turn would have had nothing for the second one, which is
the turn that matters.

The emitted text is byte for byte the plain text, checked over the whole two
turn session.

### The three places a conversation's scout has to be tidied

- **`/drop`** closes it with the session.
- **the loop's exit** closes every one, not only the current conversation's.
- **`/open`** clears it. The file has just replaced the conversation the scout's
  stream was in, and a proposer reading a conversation it is no longer in would
  be wrong without being caught: every guess is verified, so it would cost lanes
  rather than correctness — which is exactly the failure a test would not find.

`/new` needs nothing, because a new conversation is a zeroed slot and its scout
is opened on demand.

### Still not here

Sampling, which is step 4 of `TODO.md`'s entry and needs the modified rejection
rule and a proposer that carries a distribution. `--guess` refuses a temperature
in the loop exactly as it does in the single turn tasks.

---

## 0.9.5 — the output head on a grid of the activation's own, and two ideas measured out of the list

### Scope

`TODO.md`'s head entry had three routes left and this takes the largest of
them, refuses the smallest with a measurement, and closes the entry beneath it
— *stop scoring 262144 rows to pick one* — with a geometry that holds for every
clustering there is rather than for the one that was tried.

Every number below is from the reference host: four cores of a Xeon at 2.8 GHz,
AVX-512 with VNNI, the `--wide` build, four threads, the checkpoint in the page
cache, and each figure the **minimum a phase reaches over runs alternating
between the two builds**. A bare four thread sweep on this host is **28.43
GiB/s** — quote it beside any ratio below or the ratio means nothing.

### The head runs the float kernel, and now it does not have to

0.8.14 established why the output head is the slowest plane in the step per
multiply-add: `kern_level_ready` needs `sheet->enter_gain > 0`, and this export
ships `lm_head.input_activation_scale` as `0.0` where every other projection
has a real one. So the head fell to `kern_row_code`'s float spread on every
token, by construction, and every kernel change 0.8.11 through 0.8.16 made to
the integer path could never reach it.

The entry's answer was to give the head a step and take the integer path, and
to treat it as a decision about output quality rather than as a kernel change.
That is what this is.

**The step is the activation's own, per token.** `kern_level_pick` takes the
lane's largest magnitude over the 127 levels a signed byte carries, so nothing
clips and the whole range is used. `kern_level_stage` takes the step as an
argument now rather than reading it off the plane, with a flag saying which of
the two things it is: where the export calibrated the activation the caller has
already rounded onto that grid and the staging *verifies* it, exactly as
before; where it did not, the staging *rounds*. A chosen step is per lane and
not per plane — sixteen prefill lanes have sixteen different peaks — so the job
carries a list of them and the row kernels take the step from there instead of
from `sheet->enter_gain`.

Nothing else in this export is touched by it. Of its code planes only three
carry no step — the token embedding table, the per-layer embedding table, and
the head — and the first two are lookups that are never multiplied. Both towers
and all thirty-five layers are calibrated and take exactly the path they took
before.

### What it costs, and why it is then paid back

Rounding the activation onto an int8 grid is a real rounding. Over eight
ordinary prompts and forty-eight greedy steps each, the head's row disagrees
with the float path's by **0.04 root mean square and 0.49 at worst**, over the
whole vocabulary rather than at the argmax, against logits whose top runs to
twenty-seven.

Almost everywhere that is far under the gap the top token wins by. Once in 384
steps it was not: on one prompt two tokens **0.023 apart** changed places, and
the answer went a different and equally good way from there. A quantized head
that reorders a coin toss is not a quality regression — but it is a difference
a caller can see, and it does not have to be paid.

**So the sweep is taken on the grid and the decision is not.** `session_head_true`
scores the best sixty-four rows again with `kern_row_code`, the same float
product the whole plane used to take, on the unrounded activation, and writes
their exact values back. Sixty-four rows of 262144 is a four thousandth of the
plane. It is skipped where the plane brought a step from the export, because
then the levels *are* the activation and scoring a row again returns the same
number.

With it, over the same eight prompts and 384 steps:

| | integer head | and the best 64 rescored |
| --- | --- | --- |
| top token moved | 1 of 384 | **0 of 384** |
| top five reordered | 22 of 384 | **0 of 384** |
| greedy text against the float build, 96 tokens a prompt | 7 of 8 identical | **8 of 8 identical** |

The one risk the rescoring does not cover is a row *outside* the sixty-four
that should have been inside, which needs its error and the leader's to differ
by more than the gap from rank one to rank sixty-five. Over the same eight
prompts that gap never fell under **13.66 logits** against a worst error of
0.49 — a margin of twenty-eight. The tail keeps its integer values, which is
where a rounding of 0.04 is beneath a softmax's notice.

### What it is worth

| | float head | on the grid | |
| --- | --- | --- | --- |
| `final norm, head` | 7.864 ms | **5.384 ms** | **1.46x** |
| the head in multiply-adds | 51.2 G a second | **74.8 G** | |
| decode step floor | 48.38 ms | **45.70 ms** | 5.5% |
| decode | 20.67 tok/s | **21.88 tok/s** | 5.9% |
| a picture then forty tokens | 17.80 tok/s | **19.21 tok/s** | 7.9% |

The head is now **11.5% of a decode step against 16.0%**, and 5.38 ms against a
bare four thread sweep of its own 97 MiB, which is 3.41 ms. So it is at 63% of
this host's memory where it was at 43%, and what is left in the plane is worth
less than half what it was.

**And the batched head came with it, which is the speculative ceiling.**
`session_guess` verifies a block through the same `mat_mat`, so every lane of a
block now stages its own levels and the whole block takes `vpdpbusd`. On a 49 id
prompt, 128 tokens greedily:

| block | proposer | before | after |
| --- | --- | --- | --- |
| 8 | oracle | 2.09x | **2.35x** |
| 16 | oracle | 2.18x | **2.40x** |

The marginal lane at a block of sixteen is **22.76 ms to 18.08**, 21% cheaper.
`igllm guess` still reports `matches plain` on every row of the table, which is
the block path's token stream held against the plain run's, so the batch and the
sequence of steps still agree token for token.

### The broadcast, measured and refused

The entry's other kernel route was to replace three of the float loop's sixteen
instructions per sixty-four codes with one `vbroadcasti32x4` — thirteen rather
than sixteen — at the cost of a staging pass, because the four quarters come out
in the unpack's order rather than the column's.

It was built and run against the shipped loop over the real head:

| | ms | G multiply-adds a second |
| --- | --- | --- |
| the code bytes swept and nothing else | 3.411 | — |
| the shipped loop | 7.763 | 51.9 |
| the broadcast, with the activations staged | 7.880 | 51.1 |

**A wash, and it is not close.** The three instructions the route removes are
`vpbroadcastd` from memory, which retire on the load ports; the loop is bound by
ports 0 and 5, where the variable shift, the `vpermps` and the multiply-add sit,
and the route touches none of them. Twelve of those uops per sixty-four codes
before and twelve after. **Counting instructions is not counting ports**, and
this is the second time in this entry's history that the loop under test was not
the loop that was binding.

Do not reopen it. Anything that makes the float head faster has to take work off
ports 0 and 5, and the change above takes the whole loop off them instead.

### Not scoring 262144 rows: closed, and closed generally

The entry said what to measure before writing a k-means, and it was right to:
the answer costs an afternoon and it decides the whole idea.

**The bound the entry proposed prunes nothing.** A real activation out of a real
prompt, the exact logits beside it, and the rows bucketed by a sign signature
over random directions — a clustering cheap enough to be a load-time cost, which
a k-means over 262144 rows is not:

| cells | used | mean radius | radius over the mean row norm | rows the bound keeps |
| --- | --- | --- | --- | --- |
| 256 | 256 | 0.988 | 1.054 | 100.0% |
| 1024 | 1024 | 0.963 | 1.028 | 100.0% |
| 4096 | 4090 | 0.921 | 0.982 | 100.0% |
| 16384 | 15563 | 0.784 | 0.837 | 99.6% |

Sixty-four times more cells moved the radius from 1.05 of a row norm to 0.84.
The bound needs it under **0.134** — measured over 32 real greedy steps as the
top logit over `||a||`, and it ranged 0.097 to 0.201.

The plain per-row Cauchy-Schwarz bound is the floor and behaves as the entry
predicted: `||w|| ||a||` is **25.8x** the value it bounds on average, and it
keeps 262144 rows of 262144.

**And the refutation does not depend on the clustering.** A cell holding two
rows has radius at least half the distance between them, so the tightest radius
any clustering with more than one row a cell can reach is half the nearest
neighbour distance in the head. Over a sample of 256 rows against all 262144:

```
nearest neighbour distance   0.1784 to 0.9869, mean 0.8367
row norm                     mean 0.9370
best radius a cell of two    0.4184
radius the bound needs       0.1340
rows whose nearest neighbour is within twice what the bound needs: 3 of 256
```

262144 rows in 1536 dimensions are very nearly mutually orthogonal — the mean
nearest neighbour sits at 0.89 of a row norm away — so **there is no clustering
of this head into cells of more than one row whose bound can beat a top logit**,
and 1.2% of rows have a neighbour close enough that even a cell of exactly two
could ever be pruned. The entry's own rule — stop if the bound needs most of a
row to be useful — is met with room to spare, and the k-means it warned against
is not worth writing.

This closes the entry, and with it the second route of the head entry above and
step 2's remaining half in the speculative entry. What made those two worth
anything was the head's cost per byte, and the change at the top of this version
took 46% of that instead.

### A speculative round, divided

The head entry above was written against the belief that the output head is
what holds the speculative ceiling down — 0.9.0 measured the marginal lane at
13 ms and put six of them in the head, and 0.9.1 took it to three. Nobody had
divided a *round* the way `bench --verbose` divides a step, so the rest of the
lane was an assertion.

`igllm guess --verbose` now divides one. A block runs the same graph several
lanes wide, so its parts answer a different question from a step's and are kept
in a book of their own — `session_phases_block` beside `session_phases`, with
one `session_guess` counting as a round there exactly as one `session_step`
counts as a step here. The report subtracts the oracle's narrowest block from
its widest, so the difference prices the marginal lane part by part. The oracle
because it commits every lane it is given and so runs full width every round; a
proposer that draws nothing takes an ordinary step and its rounds would be a
mixture of two shapes.

On a 49 id prompt, 128 tokens, four threads, a block of 2 against a block of 16:

| part | ms at 2 | ms at 16 | ms a lane | share of the lane |
| --- | --- | --- | --- | --- |
| mlp | 36.782 | 160.924 | **8.867** | **49.3%** |
| final norm, head | 7.658 | 37.061 | 2.100 | 11.7% |
| score, softmax, blend | 4.076 | 31.862 | 1.985 | 11.0% |
| q k v | 5.597 | 21.824 | 1.159 | 6.4% |
| attn out | 5.079 | 21.037 | 1.140 | 6.3% |
| ple feed | 3.224 | 15.843 | 0.901 | 5.0% |
| named, in all | 67.377 | 319.192 | **17.987** | 100% |

**The head is no longer the ceiling.** It is 2.1 ms of 18.0 after the change at
the top of this version, and half the lane is the feed-forward. Every entry that
reasoned from "the head is half the marginal lane" should be re-read against
this table rather than re-derived.

### Where the mlp's gap goes, and three things that are not it

With the lane relocated, the mlp was measured the same way the head was — its
own 472.5 MiB, run three ways at four threads on the reference host, one quiet
run:

| | ms | |
| --- | --- | --- |
| swept sequentially | 16.4 | the memory floor |
| walked in the kernel's own pattern, no arithmetic | 18.4 | +12% |
| the kernel | 22.6 | +23% |

So the gap is two things. **Twelve percent is the row block's access pattern** —
four rows interleaved and thirty-two byte loads — which is the price of sharing
the staged levels and the epilogue between four rows, not something left on the
floor. **The rest is issue**, at about 1.4 ms per port-0-or-5 uop in the inner
loop: dropping the shift and the mask costs 19.1 ms, dropping the dot product
instead costs 21.9, and the three uops are worth about the same each. The one
that could be removed is the shift-and-mask pair, and `vgf2p8affineqb` already
removes it where the host has GFNI — 0.8.14 built that path. On a host without
it there is no single AVX-512BW instruction that shifts and masks, so this half
is the instruction set's. The same ratios hold at one thread, so none of it is
the fork, the join, or uneven threads.

And on the *batched* lane, which is the number that now matters, three
explanations were measured and all three came back small:

- **The per-lane epilogue**: `kern_row_code_level_many` closes every row for
  every lane on its own, 7.7 M times over the mlp at sixteen lanes, where the
  one lane path has folded sixteen rows into a vector since 0.8.12. Replacing
  the whole close with a raw store is worth 3.7 to 8.7% — 0.3 to 0.8 ms of a
  9.3 ms lane.
- **The staging stride**: every lane's levels were laid down `level_limit`
  apart, the widest code plane the model binds, so sixteen lanes of a 1536
  column plane sat 8960 bytes apart where the 24 KiB they read would have fitted
  in the first level cache side by side. Staging at the plane's own width is
  6.6% in isolation. **End to end on this host it is a wash inside the noise**,
  and it ships because it is the same work with better locality and costs
  nothing — not because it was measured to pay.
- **Ports**: five port-0 uops per four lanes per block is about 2 ms a lane over
  four threads, against 8.9 measured.

Four fifths of the batched lane is therefore still unexplained, and `TODO.md`
says so and says to take a hardware counter to it rather than another loop.

### What moved in the source

- `kern_level_pick`, `kern_level_round`: the step a lane chooses, and the
  rounding onto it.
- `kern_level_ready`: asks about the shape of the plane and no longer about the
  step, because the step is not always the plane's.
- `kern_level_stage`: takes the step and a flag saying whether to verify the
  levels or round them.
- `kern_row_code_level`, `..._rows`, `..._wide`, `..._many`: take the step
  rather than reading `sheet->enter_gain`; the many lane form takes one per
  lane out of `job->step_list`.
- `back_level_pick`: whether a plane will bring a step of its own, which is what
  says a row is worth scoring again.
- `session_head_true`: the best sixty-four rows of the head, on the float path.
- `session_phases_block`, `session_phase_clear`, and a second `app_phase_book`
  in the session with `phase_ref` naming the one the timer is filling — so a
  round is divided without a block's parts landing in a step's buckets.
- `main_guess_phases`: the round division and the marginal lane, on
  `guess --verbose`.
- `back_mat_mat`: the level staging laid down at the plane's own width.

### Tests

`level` grew two cases and had one rewritten. The rewrite is the interesting
one: "a step the activations are not on changes nothing" used to take its
baseline by leaving `enter_gain` at zero, and zero no longer means the float
path — it means a step chosen from the activation. The baseline is now taken
with the integer path switched off at the backend, which is what it was always
trying to say. The two new cases are the chosen step at every lane count
against the dense product, each lane with a peak of its own so a shared step
would fail, and `kern_level_pick` itself on the peak, on zero, and on a value
that is not finite.

797 pass on the wide build, 788 on a build without the integer dot product,
where every plane takes the float path exactly as it did before.

---

## 0.9.6 — a block of queries through the tower's attention, at the same arithmetic

### Scope

`TODO.md`'s vision entry — *the tower's own attention, tiled, with the
normalizer carried* — opened by asking for a fresh profile before any schedule
was written, because 0.8.9 had changed the shares 0.8.8 recorded. This takes
that profile, finds the prize where the entry said it would be, and spends it
on something cheaper than the tiling the entry proposed: a block of queries
rather than a block of keys, which needs no running maximum, no normalizer
carried across tiles, and no change to the order of any sum.

Every number below is from the reference host: four cores of a Xeon at 2.8 GHz,
AVX-512 with VNNI and no GFNI, the `--wide` build, four threads, the checkpoint
in the page cache. The picture is 768 by 768 at the full patch budget, which is
2304 patches through 16 layers of width 768 and 12 heads.

### The profile, which is the entry's own first step

A picture divided into its named parts, one tower forward, four threads:

| part | s | share |
| --- | --- | --- |
| **score, softmax, blend** | **5.44** | **52.7%** |
| feed-forward | 2.63 | 25.5% |
| q k v | 1.34 | 13.0% |
| attn out | 0.46 | 4.4% |
| head norms, rotary | 0.26 | 2.5% |
| gather k v | 0.08 | 0.8% |
| named, in all | 10.33 | 100% |

So the entry's premise holds and is if anything understated: the projections
took the integer path in 0.8.9 and the attention did not, and it is now over
half of a picture. Split further, scoring is 51% of that phase, the blend 43%
and the softmax 6.6%.

**Both are far off this host's arithmetic, which says it is not a memory
schedule that is missing.** Scoring runs 65.2 G multiply-adds in 9.75 thread
seconds and the blend the same 65.2 in 8.18 — 6.7 and 8.0 G a second against a
256-bit FMA peak of 44.8 a core. A gathered head is 590 KiB of keys and 590 KiB
of values, and the two are walked in separate loops, so each sits inside this
host's 1 MiB of private second level cache for the whole of the band. Neither
loop is waiting on memory; both are waiting on their own shape.

### What the shape was

`tower_attend_band` took one query at a time. For that query it scored against
every one of the 2304 gathered keys — 2304 calls to `kern_dot_real`, each of
them eight multiply-adds over sixty-four columns followed by a horizontal
reduction into a float and a scalar store — softmaxed the row, and blended.
Then it read the same 590 KiB of keys again for the next query, and again for
the one after that.

Two costs, and the second is the larger. **The run is read once per query where
it could be read once per block of queries**, and **a sixty-four column dot
pays a close that costs about as much as the eight multiply-adds it closes**.

### The block, and why it is not the tiling the entry asked for

`RESEARCH.md` idea 11 is flash attention's schedule: tile the keys, carry a
running maximum and normalizer across tiles, accumulate the blend, and never
materialize the score matrix. Its prize is the score matrix, which at 2304
patches is 2304 by 2304.

**This engine never materialized that matrix in the first place.** A band holds
one score row, not the grid, so the memory the tiling would save was already
saved and what is left of the idea is the summation-order change it costs. The
axis that is actually loose is the other one: the queries, which share every
byte both loops read and shared nothing.

So the block is four queries wide. `kern_score_block` loads a key row once and
multiplies it into four queries' accumulators; `kern_blend_rows_many` loads a
value row once and folds it into four queries' blocks of running sums. The
softmax between them stays per query, because that is what it is.

**Four, and not more, for the reason `KERN_CODE_LANE` is four.** Four queries
against a thirty-two column block of values is sixteen live accumulators, which
is every vector register an AVX2 host has; a host with AVX-512's thirty-two was
measured at the same width and does not need more to cover the multiply-add's
latency.

### The arithmetic is the one lane path's, and that is checked rather than argued

Neither kernel reassociates anything.

**Scoring.** Each lane keeps the same two accumulators over the same slots in
the same order `kern_dot_real` keeps them in. The close is the only thing that
changes shape: `kern_dot_total` adds a vector's two halves and then pairs what
is left twice, and a pair of `_mm_hadd_ps` over four lanes at once pairs
exactly the same four floats in exactly the same order — `(a0+a1)+(a2+a3)` —
for four lanes in one lane's worth of instructions.

**The blend.** Value `v` still takes span 0 first and span `span_count - 1`
last, into an accumulator of its own. The multiply and the add stay a multiply
and an add: **the fused multiply-add is deliberately not taken here.** It is
the obvious next instruction, it measured 2.2x against the shipped loop where
the multiply and add measured 1.4x, and it is refused anyway, because it drops
the intermediate rounding and a picture's rows would stop being the rows that
ship. It is also the *worse* kernel on a plain AVX2 host — 1.28x against 1.32x
— where sixteen accumulators and four value registers do not fit in sixteen
registers and the fused form spills. The 0.6x left on the table is written into
`TODO.md` as a decision someone can take later with their eyes open.

Two cases in the `kernel` group hold both kernels to the one lane kernels for
**equality and not for nearness**, on shapes chosen to be awkward: a span count
that is not a multiple of sixteen, a row count that is not a multiple of
anything, and a lane count of six, which runs one full block and a two lane
tail through the one lane path. Swapping the blend's multiply-and-add for the
fused form fails the second case, which is what it is there for.

### What it is worth

The phase, over runs alternating between the two builds in both orders:

| part | before | after |
| --- | --- | --- |
| **score, softmax, blend** | **5.26 – 5.61 s** | **2.72 – 3.01 s** |
| tower, named parts in all | 10.05 – 11.08 s | 6.67 – 8.53 s |

**The phase is about 1.9x** and the ranges do not overlap. End to end, a
picture through the tower on the uninstrumented builds is **10.64 s to 8.10 s**
— minimum of three alternating runs each, less the 0.44 s the model takes to
map — and the same measurement on a 512 by 384 picture and a 896 by 896 one
gives 1.40x and 1.37x on the whole run including the load.

Note what the first table's other rows do *not* say. `q k v` and the
feed-forward appeared to move by five to ten percent between the two builds in
one direction, and the control — the same series run with the order of the two
builds reversed — put the first run of *either* build ahead of the runs after
it. That is the host drifting over a series, not the change, and neither part
is touched by it.

The text stack is not affected at all: the towers are not in the token loop,
and decode measures the same on both builds.

**The output is byte for byte what it was.** `logits` after a picture, after a
clip, after both, and after neither is byte-identical between the two builds on
every size tried, and 799 tests pass on the wide build, 788 on a build with no
integer dot product, and 788 on an SSE2 build.

### Where it leaves the entry

Scoring and the blend are now 34 to 41% of a picture against 52.7%, and the
feed-forward has become the largest part of a tower. What is left in the
attention is a phase running at roughly twice the multiply-adds a second it
was, which is still well short of this host's FMA peak — and the query axis,
which was the loose one, is now spent. The remaining routes are in `TODO.md`.

### Code

- `KERN_GRID_LANE`, `kern_score_block`, `kern_blend_rows_many`: the two block
  kernels, beside `kern_blend_rows` which they are a block of.
- `tower_attend_band`: a block of queries a turn, with the softmax still per
  query between the two halves.
- `tower_room_open`: a band's score scratch is a block of rows rather than one.

### Tests

The `kernel` group grew the two equality cases described above. 799 pass on the
wide build, 788 on a build without the integer dot product and 788 on SSE2.

---

## 0.9.7 — a picture's rows kept against the picture

### Scope

`TODO.md`'s *reuse a picture's rows when it is the same picture* — `RESEARCH.md`
idea 12. Exact, cheap and narrow, and worth more after 0.9.6 than before it only
in the sense that a picture is now 8 s rather than 10.6: asking three questions
about one photograph is the ordinary case, and until now every one of them paid
the whole tower.

**This is not fresh-image acceleration and must not be reported as one.** A
picture the engine has not seen costs exactly what it cost before. What changes
is the second time.

### What is kept, and where

The **post-projector rows** — `soft_limit` by the text stack's width, 256 by
1536 on the shipped export, 1.5 MiB a picture. Those are the last thing the
vision path produces and the first thing the text stack consumes, so keeping
them skips the resize, the patch cut, all sixteen encoder layers, the pooling
and the projector at once, and leaves the caller holding exactly the
`app_media` it would have held.

It lives on the **model** and not on a session. The rows are the weights' answer
to a picture rather than a conversation's, so a second conversation about the
same photograph has them too — which is the case `--keep` and `chat --loop`
cannot reach, because both of those match on rows the tower has already been run
to produce. Like everything else hanging off a model it assumes one caller at a
time, which is the assumption `pool_group` already makes.

Four pictures, fixed, least-wanted evicted. That is the working set the ordinary
case has — a photograph and some questions, or a handful of pictures compared
against each other — and depth beyond a working set does nothing for it. The
1.5 MiB an entry costs is reported without being asked for, because it comes out
of the engine's own allocator: `bench` with a picture goes from 2205.5 MiB
allocated to 2207.0.

### The identity, which is the whole of the risk

A cache like this has exactly one failure mode that matters: handing back rows
for a picture the caller never showed, silently. So the identity is the part
that got the attention.

**It is taken on the decoded raster**, after `image_read` and before anything
else, so the container and the decoder are out of it: the same photograph as a
png and as a lossless bmp is one entry, a lossy re-encode of it is not, and both
of those are the right answer. Reading and decoding the file is still paid on a
hit; what is saved is the tower, which is all but the whole of what a picture
costs.

**Into it go** every decoded sample, the raster's own shape, and the tower
configuration that decides what the samples become — the patch and pool sizes
and the soft token budget, which together fix the resized grid and the pooling
geometry; the tower's depth and width; the text stack's width, which is what the
projector lifts into; and the checkpoint's mapped size, the way `keep_mark` uses
it, so two exports of the same shape are not one identity.

**Two independent mixes, not one.** Sixty-four bits puts a birthday collision
somewhere around four billion pictures, which is not a number to rest a silent
wrong answer on. The samples go through an FNV-1a and through `keep_mix` with a
different seed at the same time, which is a hundred and twenty-eight bits, and
the raster's shape is compared exactly beside them, so a collision has to agree
on that as well. Over a 6.8 MiB raster both mixes together measure **2.5 to 5.1
ms**.

**The backend is deliberately not in it, and that is written down rather than
left to be noticed.** This store is never put in a file, so a backend cannot
change under an entry — it is fixed for the life of the process that made it.
Anything that gives these rows a life beyond one process has to put the backend
and an encoder version in first.

### What it is worth

The case the entry names, end to end: a photograph, a question about it, `/new`,
the same photograph, another question — **53.53 s to 39.72 s**, and the two
conversations are byte for byte identical. The same picture twice in one prompt
is 9.12 s against 9.18 s for one; two different pictures is 17.19 s, which is
both towers, as it should be.

A miss costs the identity and the copy. Over three alternating runs a single
picture measured 8.79 s without the store and 8.94 with — under 2%, and the
identity is 5 ms of it.

`logits` after a picture, a clip, both and neither is byte for byte what it was
on every case tried, and a hit hands back the tower's rows bit for bit.

### Code

- `MEDIA_KEEP_COUNT`, `media_note`, and `vision_note` / `media_turn` on
  `app_model`: the store.
- `media_mark`, `media_recall`, `media_keep`, `media_keep_free`.
- `media_image`: the identity taken on the decoded raster, the store consulted
  before the tower and written after it.
- `keep_mix` moved up the file, so the store and `keep_mark` share one mixer.

### Tests

Six cases in the `tower` group. The store's arithmetic is held for **equality**
— a hit hands back the tower's rows bit for bit, and copied out rather than
lent, so the caller owns them either way. The identity is held three ways: one
sample moved is a different identity in *both* mixes, the same samples are the
same identity whatever the call around them, and another picture gets rows of
its own.

The eviction cases ask `media_recall` directly rather than going through
`media_image`, and that is the point rather than a convenience: **the tower is
deterministic, so a miss that re-runs it hands back exactly the rows a hit would
have.** A test written on the rows passes with the store emptied between every
call — the first draft of this one did, and it was rewritten when setting
`MEDIA_KEEP_COUNT` to 1 failed to fail it.

809 pass on the wide build, 800 on a build without the integer dot product and
800 on SSE2.

---

## 0.9.8 — the block verified under a temperature, by the rule that makes it exact

### Scope

The speculative entry's step 4, which had been open since 0.9.0 and read: *so
either the feature is greedy-only and says so, or it needs a proposer that
carries a distribution. Decide which before plumbing it into `chat`.*

It turns out to be a false choice, and that is the whole of this version. A
proposer needs a distribution only because the rejection rule divides by it —
and the scout's distribution is a **point mass**, which is a distribution like
any other and the easiest one to divide by.

`--guess` was refused outright above `--heat 0`, and the default taste is
`--heat 1`, so the flag was unusable without being asked for explicitly. It now
works under any taste.

### The rule, and why it needs nothing the scout does not have

Speculative sampling accepts a proposal `t` with probability `min(1, p(t)/q(t))`
and, on a rejection, draws from the residual `norm((p - q)+)`. The scout names
one token and nothing else, so `q(t) = 1` and `q` is zero everywhere else. Put
that in:

- accept with probability `min(1, p(t)/1)`, which is **`p(t)`**;
- on a rejection the residual is `p(x)` for every `x` other than `t`, and
  `max(0, p(t) - 1) = 0` at `t` itself — so it is **`p` with the guess taken out
  and what is left renormalized**.

Both halves are exactly computable from `p` alone. There is nothing to
approximate and no second model to carry, and the token that comes out is drawn
from `p` exactly.

**`p` is the caller's whole taste**, not a bare softmax: heat, the echo penalty,
top-k and top-p, all of it. So `session_pick_at`'s body is lifted out into
`pick_shape`, which builds the shaped distribution, and `pick_draw`, which takes
a token out of one. Both paths now sit on one definition of what a taste means,
and the sampled text of a plain run is byte for byte what it was — checked on
three seeds.

**Each lane is shaped against its own history.** Lane `j` is judged as a plain
run would judge it having committed the first `j` guesses: `guess_echo + j + 1`
echo entries and cache position `guess_from + j + 1`, neither of which is what
the session holds while the whole block is in flight. Reading the echo history
off the session instead is the kind of mistake that changes a distribution
slightly and silently; there is a test below that fails on it.

**The draws are derived rather than sequential.** `session_pick_at` reseeds from
`seed ^ fill_count` and draws once, which fits a path that takes one draw per
position. A block fits it twice over — its lanes share one `fill_count`, and a
lane that rejects takes two draws rather than one — so a block derives each draw
from the seed, the position, and which of the two it is, through splitmix64's
finalizer. The finalizer is not decoration: `draw_next` is an xorshift64, and
xorshift's first word out of two nearly equal seeds is nearly equal too, so the
accept draw and the resample draw would otherwise be a constant apart.

### What it promises, and what it does not

A greedy block is byte for byte a greedy run, and 0.9.8 does not touch that —
`--heat 0 --guess 4` and `--guess 8` are identical to 0.9.7's.

**A sampled block is not, and cannot be.** A round takes one draw where its
guess is accepted and two where it is not, so the block path and the plain path
consume randomness at different rates and diverge from the first rejection: the
same seed gives a different stream at a different `--guess`, and the same stream
only at the same one. What is equal is the **distribution** the stream is drawn
from, which is the guarantee speculative sampling makes and the only one it
makes. This is stated in the README rather than left for someone to discover.

### What it is worth

Acceptance under a temperature is `p(t)` — the model's own probability of the
guessed token — so it tracks how certain the continuation is, where greedy
acceptance only asks whether the guess was the argmax. That cuts both ways, and
both ways were measured on the reference host at four threads, `--heat 1` with
the default top-k 64 and top-p 0.95:

| workload | plain | `--guess 4` | `--guess 8` |
| --- | --- | --- | --- |
| repeat a passage back verbatim | 33.55 tok/s | **68.32** (2.04x) | **79.15** (2.36x) |
| free generation | 33.99 | 31.92 (0.94x) | — |

On the quoting workload it keeps **100% of 63 guesses** at a block of four and
96% of 77 at eight, and 2.04x and 2.36x are *better* than the greedy path's
1.69x and 1.80x on its own quoting prompt — because verbatim repetition is
where the model is nearly certain, and near-certainty is exactly what this
acceptance rule rewards. Where the model is not certain it rejects, so a
paraphrasing prompt sits between the two rows above.

Free generation costs about 6% over three alternating runs, which is the same
place greedy sits (0.92 to 0.95), and for the same reason: the scout draws
nothing at all most rounds, and a round it draws nothing in is an ordinary step.
So `--guess` stays a flag rather than a default under a temperature, exactly as
it is without one.

### Code

- `pick_shape` and `pick_draw`: `session_pick_at`'s body, lifted out whole so a
  second caller can have the distribution rather than a token from it.
- `session_guess_taste`: the rejection rule, public beside `session_guess`.
- `guess_draw_state`, `GUESS_DRAW_TAKE`, `GUESS_DRAW_MAKE`: a block's draws.
- `pick_share`: what a shaped distribution gives one token, zero where the taste
  has already cut it.
- `main_answer_guess`: the rule where the taste has a temperature, the argmax
  comparison where it does not, and a plain step drawn with the caller's taste
  where the scout proposes nothing.
- `main_serve`: the refusal above `--heat 0` is gone.

### Tests

The `guess` group grew six cases, and the first is the one that matters.

**The theorem is held as a theorem.** The shaped distribution at lane 0 is
computed directly, then a hundred thousand rounds are run through the rejection
rule with a different seed each and the tokens they commit at position 0 are
counted. What is being checked is that the mixture `p(t)*[t] + (1 - p(t))*residual`
**is** `p`. A hundred thousand rounds over twenty-seven tokens puts a bucket's
standard error under 0.0016, so the bar of 0.01 is six sigma — wide enough never
to flake and narrow enough that no wrong rule fits under it. Two of them were
built and neither does: resampling without removing the guess fails it, and
accepting whenever the model gives the guess any mass at all fails it twice.

It runs **twice**, once on a bare softmax and once with top-k, top-p and the
echo penalty on. The second is what holds the per-lane history, and it was added
because the first draft passed with the history read off the session — the
penalty was switched off, so there was nothing for the mistake to move. The
guess is now chosen out of the shaped row rather than taken from the fixture,
because an arbitrary token is usually one top-k has already cut, and a guess
that can only ever be rejected exercises half the rule.

Three more hold the edges and the boundary: a guess the taste has cut is never
accepted, a guess holding every slot the taste kept is always accepted and never
divides by an empty residual, and a greedy taste is refused rather than given an
invented rule.

822 pass on the wide build, 813 on a build without the integer dot product and
813 on SSE2.
