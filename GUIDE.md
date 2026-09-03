# GUIDE

A complete tour of the igllm implementation: what each file holds, how a
prompt becomes a token, which formats are read, how the kernels are built,
and where an accelerator would attach.

---

## 1. File map

| file          | role                                                         |
| ------------- | ------------------------------------------------------------ |
| `app_core.c`  | the engine: a public interface followed by ten layers        |
| `app_main.c`  | the command line front end                                   |
| `app_test.c`  | the unit tests                                               |
| `app_test.py` | parity and throughput comparison against transformers        |
| `app_fake.py` | builds a synthetic checkpoint and quantizes it               |
| `app_diff.py` | layer by layer comparison against transformers               |
| `run.py`      | install, build, test, check, parity, run, clean workflows    |

`app_core.c` is a single translation unit. `app_main.c` and `app_test.c`
each `#include "app_core.c"`, so a build is one compiler invocation per
binary with no archive, no header search path, and no build system. The
file is split by two include guards:

- `APP_CORE_INCLUDED` wraps the public interface — the types and the
  function declarations a caller needs.
- `APP_CORE_IMPLEMENTED` wraps everything else — private to the engine.

## 2. Naming

Every name is `noun_verb` or `noun_noun`, with the module as the first word.
`model_load` and `model_free`, `session_open` and `session_close`,
`token_encode` and `token_decode`, `plane_row` and `plane_gain`. Struct
fields follow the same rhythm: a `_count` is a quantity, a `_size` is a
dimension, a `_limit` is a cap, a `_list` is an array, a `_room` is scratch,
a `_flag` is a boolean, a `_sheet` is a weight matrix, and a `_span` is a
view into memory the engine does not own.

## 3. Layers

### 3.1 Public interface

Six types and twenty-odd functions:

- `app_code` — the single result enum; zero is success.
- `app_setup` — thread count, context length, verbosity.
- `app_taste` — temperature, top-k, top-p, repetition penalty, seed.
- `app_tally` — prefill and decode timings, token counts, memory.
- `app_model` — an opaque loaded checkpoint, shared and read-only.
- `app_session` — an opaque conversation, one per concurrent stream.

The split matters: a model is loaded once and can back many sessions,
because all mutable state — the key and value caches, the scratch arenas,
the sampler — belongs to the session.

### 3.2 Platform layer

Thin shims, no third party code:

- `mem_alloc`, `mem_clear`, `mem_free` — aligned allocation, so vector loads
  never straddle a page in a way the host dislikes.
- `file_map`, `file_open`, `file_close` — `mmap` on POSIX,
  `CreateFileMapping` and `MapViewOfFile` on Windows. Weights are never
  copied; the operating system pages them in on demand.
- `file_slurp`, `file_exists`, `path_join` — small text and path helpers.
- `time_now` — `clock_gettime(CLOCK_MONOTONIC)` or `QueryPerformanceCounter`.
- `host_thread_count` — `sysconf` or `GetSystemInfo`.
- `pool_group` — a fork-join thread pool: `pool_open`, `pool_run`,
  `pool_close`. `pool_run` hands the same function to every worker with a
  slice index, waits, and returns. There is no queue and no allocation in
  the token loop.
- `slice_span` — divides a range into bands, spreading the remainder over
  the leading bands so no worker is more than one element behind.

### 3.3 JSON layer

An arena document reader. `json_read` parses into `json_tree`: one flat
array of `json_node` records plus one text arena. Nodes reference each other
by index, not by pointer, so the tree is one allocation pair and freeing is
two calls. Strings are decoded once, including `\uXXXX` escapes and
surrogate pairs, into UTF-8.

Reading is by name or slot: `json_field`, `json_item`, `json_count`,
`json_number`, `json_text`, and the convenience trio `json_field_number`,
`json_field_text`, `json_field_flag`, each taking a spare value for absent
fields. This is what reads `config.json`, `generation_config.json`, the
safetensors headers, and `tokenizer.json`.

### 3.4 Store layer — safetensors

A safetensors file is an eight byte little-endian header length, a JSON
header, then the payload. `store_open` handles both layouts:

- `model.safetensors.index.json` present — every shard named in
  `weight_map` is opened and merged into one name space.
- otherwise — `model.safetensors` alone.

Each entry becomes a `store_span`: dtype, rank, shape, and a pointer into
the mapped payload. `store_find` resolves a name through an FNV hash and
open addressing. `store_type_of` recognises `F32`, `F16`, `BF16`, `I8`,
`U8`, `I16`, `I32`, `I64`, and `BOOL`.

Nothing is read eagerly. A tensor costs address space, not resident memory,
until it is touched.

### 3.5 Quant layer

Two layouts are read, told apart by `quant_method` in `quantization_config`.

**The Gemma layout**, which is what the shipped checkpoint uses. A quantized
linear is its `weight` and a `weight_scale`; a quantized embedding table is an
`embedding_quantized` and an `embedding_scale`. The codes are ordinary bytes:
four to a byte at two bits, two at four bits, one at eight. The scale carries
one column per group — one for a per-channel plane, thirty-five for the
per-layer embedding table, one to each layer's block. Two and four bit codes
are unsigned with an offset of half the range; eight bit codes are signed
`I8` with no offset.

Nothing in such a tensor records how wide the row was, because the stored
shape *is* the packed width, so `plane_bind` is told the input width its
caller expects and takes the bit width that makes the byte count come out
right. `q_proj` reads the state, `down_proj` reads what `gate_proj` wrote,
`o_proj` reads what `q_proj` wrote. A hint no width explains is refused
rather than decoded. The alternative, a regular expression engine for the
`module_quant_configs` map the configuration carries, would be a few hundred
lines that could only ever agree with the shapes.

**The compressed-tensors pack-quantized layout**, which the synthetic
checkpoints use. A quantized linear is four tensors:

| tensor              | meaning                                          |
| ------------------- | ------------------------------------------------ |
| `weight_packed`     | `int32` words holding a dense bit stream of codes |
| `weight_scale`      | one scale per output row and input group          |
| `weight_zero_point` | optional, packed along the row axis               |
| `weight_shape`      | the logical `[out, in]` shape                     |

The packing is dense and little-endian: element *i* of a row occupies bits
`[i*bits, i*bits + bits)`, and rows are padded up to a 32-bit boundary. On a
little-endian host the mapped bytes are usable **as they are** in either
layout — `pack_read` walks the bit stream directly with no repacking pass and
no second copy of the weights, and a byte-packed row at two, four or eight
bits is the same stream. Codes carry a sign offset, so the dequantized value
is

```
w = ((code ^ flip) - offset - zero_point) * scale
```

where `flip` is the offset for a signed byte plane and zero otherwise — a
signed byte being the offset code with its top bit turned over, one xor puts
both conventions through the same decode.

`plane` is the storage record for any weight. It is either `PLANE_REAL` — a
plain `f32`, `f16`, or `bf16` matrix read in place — or `PLANE_CODE` — either
packed form above. `plane_row` decodes a whole row when a row is what is
wanted, for example a single embedding lookup. `plane_bits_of` recovers the
bit width from the shape and the packed word count, so a checkpoint that
mixes widths across tensors, as this one does, is handled without trusting
the configuration.

**Activation ranges.** The Gemma export calibrates two more numbers for every
quantized projection, an `input_activation_scale` and an
`output_activation_scale`, and the reference rounds what goes into the
projection and what comes out of it onto those grids at eight bit levels.
`quant_step` does that, `session_lift_many` applies it on both sides of every
code plane, and the rounding is to even: a product of two quantized planes is
an exact multiple of the two steps, so the value being rounded lands on a half
step often rather than never, and rounding up instead moves whole activations
by a whole step. A step of zero means the layer was never calibrated and
nothing happens.

`quant_rule` and `quant_book` carry what `quantization_config` declares:
bit width, group size, symmetry, strategy, and the ignore list. When the
configuration declares quantized input activations, `quant_act` reproduces
the reference fake-quantization per token before the matrix product. That is
the dynamic rule, which finds its range from the activation; `quant_step` is
the static one, which is told it.

### 3.6 Kernel layer

Every kernel has a plain scalar definition and, where it pays, a vector
path chosen at compile time behind one macro layer: `APP_SIMD_AVX2`,
`APP_SIMD_SSE2`, `APP_SIMD_NEON`, or none. `back_flavor` names the path that
was selected.

- `kern_dot_real` — dot product against an `f32`, `f16`, or `bf16` row.
- `kern_dot_code` — dot product against a packed row, specialized for the
  two, four, and eight bit cases and general otherwise. The two and four bit
  cases have vector paths on all three targets: a nibble or a quarter byte
  unpacks with whole-vector shifts and masks. Both require the group to start
  on a byte boundary and fall back to the bit-stream loop when it does not.
- `kern_row_code` — one output row of a quantized matrix, formulated as

  ```
  row = Σ_groups gain * ( Σ_j a[j]*code[j]  -  (offset+zero) * Σ_j a[j] )
  ```

  The per-group activation sums are computed once per matrix product by
  `kern_group_sum` and shared by every row. This removes one subtraction
  per element from the inner loop and accumulates in a wider range.
- `kern_row_code_many` — the same row against several activation vectors at
  once. A group of codes is unpacked into a small float scratch and dotted
  against every lane, so a batch pays the decode cost of a single vector.
- `kern_mat_vec_band` — one band of rows, the unit of work given to the pool.
  It carries a lane count, so the same band function serves a matrix-vector
  product and a matrix-matrix product.
- `kern_norm_rms` — `x * rsqrt(mean(x²) + eps) * weight`. Gemma 4 uses the
  weight directly, **not** `1 + weight`.
- `kern_gelu_tanh`, `kern_gelu_gate` — the tanh approximation, and the gated
  form the MLP wants.
- `kern_soft_max` — streaming maximum then streaming sum, so a long attention
  row never overflows.
- `kern_rope_turn` — the rotate-half rotary transform, in place.
- `kern_add`, `kern_scale` — residual and scalar helpers.

### 3.7 Backend table

All compute is reached through `back_desk`, a record of function pointers
plus the scratch it needs:

```c
typedef struct back_desk {
  const char *name_text;
  pool_group *pool_ref;
  float      *sum_room;
  int         sum_limit;
  void (*mat_vec)(struct back_desk *, const plane *, const float *, float *);
  void (*mat_mat)(struct back_desk *, const plane *, const float *, int, int,
                  float *, int);
  void (*norm_rms)(struct back_desk *, const float *, const float *, int, float, float *);
  void (*soft_max)(struct back_desk *, float *, int);
  void (*gelu_gate)(struct back_desk *, float *, const float *, int);
  void (*rope_turn)(struct back_desk *, float *, int, const float *, const float *);
} back_desk;
```

`mat_vec` is `mat_mat` with one lane, so there is a single implementation to
replace. `back_open` binds the CPU implementation. The model and session layers never
call a kernel by name — they call `desk->mat_vec` and its siblings. A GPU or
NPU backend therefore needs three things and touches nothing else:

1. its own `back_open` that fills the table with device entry points,
2. a device buffer handle beside each `plane`, filled while the loader is
   already walking every tensor,
3. a residency decision per tensor, which the loader already makes when it
   chooses between `PLANE_REAL` and `PLANE_CODE`.

The loader deliberately keeps "the bytes of a weight" separate from "the
handle to a weight", which is what makes that substitution local.

### 3.8 Model layer

`config_read` fills `model_form`, the description of everything that affects
the arithmetic:

| field                          | meaning                                    |
| ------------------------------ | ------------------------------------------ |
| `vocab_count`, `state_size`    | vocabulary and hidden width                |
| `inner_size`, `layer_count`    | MLP width and depth                        |
| `head_count`, `kv_count`       | attention and key-value head counts        |
| `head_size`, `whole_head_size` | head widths, sliding and full attention     |
| `slide_span`                   | sliding window length                      |
| `ple_vocab`, `ple_size`        | per-layer embedding table and width        |
| `share_count`                  | trailing layers that reuse another cache   |
| `twin_flag`                    | `attention_k_eq_v`                         |
| `wide_flag`                    | `use_double_wide_mlp`                      |
| `moe_flag`, `expert_count`     | `enable_moe_block` and the expert count    |
| `expert_top`, `expert_inner`   | experts kept per token, and expert width   |
| `norm_eps`, `logit_cap`        | RMSNorm epsilon, final logit softcapping   |
| `kind_list`                    | one mark per layer: sliding or full        |
| `rope_list`                    | one rotary table per layer kind            |

`layer_wing` then resolves the per-layer settings **once**, at load, so the
hot loop is branch-light: head size, key-value head count, group share, the
cache span, whether the layer shares another layer's cache and from where,
whether it must keep a full-length cache for a sharer, and pointers to every
weight the layer uses.

Two properties are derived from the tensors themselves rather than trusted
from the configuration — the head size, from the row count of `q_proj`, and
the key-value head count, from the row count of `k_proj`. So is the expert
width, from half the row count of the stacked `gate_up_proj`. A checkpoint whose
configuration disagrees with its weights still loads correctly.

Rotary tables. `rope_build` produces one table per layer kind. Sliding
layers use the default schedule over the whole head. Full-attention layers
use the proportional schedule: only the first `partial_rotary_factor *
head_size / 2` pairs receive a frequency, and the remainder are left at zero
so their rotation is the identity. `rope_wave` evaluates the angles for one
position on demand rather than materializing a table for the full context,
which would cost hundreds of megabytes at the model's maximum length.

`model_prefix_pick` detects the weight name prefix — `model.language_model.`,
`language_model.model.`, `model.`, or none — so checkpoints exported by
different transformers versions load without a flag. It probes each candidate
for `embed_tokens.weight`, `embed_tokens.weight_packed` and
`embed_tokens.embedding_quantized` in turn, since the name of the embedding
table is what distinguishes the quantization layouts as much as the prefix
distinguishes the exporters.

### 3.9 Token layer

`token_load` reads `tokenizer.json`: the vocabulary, the merge list, the
added tokens with their special marks, and the metaspace behaviour from the
normalizer and pre-tokenizer records. Byte-fallback entries of the form
`<0xNN>` are indexed into a 256 entry table.

`token_split` maps text to UTF-8 runes, replacing spaces with the metaspace
mark and prepending the mark when the tokenizer asks. `token_encode_book`
then applies the standard byte-pair merge loop, always taking the lowest
ranked adjacent pair, and falls back to byte tokens for anything the
vocabulary does not hold. `token_decode_book` reverses that, turning the
metaspace mark back into a space, expanding byte tokens, and emitting
nothing for control tokens.

`token_frame` applies the Gemma chat frame for instruction-tuned prompts:
the sequence marker, then a user turn, then the opening of a model turn.
Gemma 4 writes the turn markers `<|turn>` and `<turn|>`, and the older
`<start_of_turn>` and `<end_of_turn>` are probed behind them, since nothing
else in a checkpoint distinguishes the two vocabularies. `chat` and `logits`
both frame; `--raw` and every other task do not.

### 3.10 Session layer

A session owns the caches and the scratch. `session_open` allocates once,
sized from the model, and nothing in the token loop allocates again.

Key and value caches are sized per layer. A sliding layer needs only its
window and uses a ring, indexed by `position % cache_span`. A full-attention
layer, and any layer that is the source for a sharing layer, keeps the full
context length. Layers that share simply read the source layer's cache and
never write their own. That sizing is what keeps the cache in the tens of
megabytes rather than the better part of a gigabyte.

Every scratch buffer holds `KERN_LANE_LIMIT` lanes with a named stride, and
the whole graph carries a lane count. Decode is a batch of one, so there is
one code path rather than two.

`session_pass` runs a batch of tokens:

1. embedding lookup, scaled by `sqrt(hidden_size)`
2. the per-layer embedding block: table lookup scaled by `sqrt(ple_size)`,
   the model projection scaled by `hidden_size^-0.5`, normalized, and mixed
   with the token identity at `2^-0.5`
3. for each layer, `session_layer`:
   - `enter_norm` → attention → `after_attn_norm` → residual add
   - `before_feed_norm` → gated MLP → the mixture branch, if the checkpoint
     has one → `after_feed_norm` → residual add
   - the per-layer embedding gate → `after_ple_norm` → residual add
   - a scalar layer gain
4. the final norm, the output head, and optional logit softcapping, for the
   last token of the batch only

`session_attend` is the attention itself. Queries and keys are normalized
with their own RMSNorm weights, rotated, and scored against the cache.
Attention scaling is `1.0`, not `1/sqrt(d)`; the query and key norms absorb
it, which is what the reference does. Values carry a norm without a scale.
When `attention_k_eq_v` is set, a full-attention layer has no `v_proj` and
uses the raw pre-norm key projection as its values.

When `enable_moe_block` is set, the dense MLP above is the shared expert and
a routed branch runs beside it. `session_route` normalizes the pre-MLP
residual without a scale, multiplies by `router.scale * hidden_size^-0.5`,
softmaxes the projection to one score per expert, keeps the top *k*,
renormalizes the kept weights to sum to one, and scales each by its
`per_expert_scale`. `session_expert` then runs the selected experts and
blends them. The two branches carry `post_feedforward_layernorm_1` and
`post_feedforward_layernorm_2` respectively and are summed before the shared
`post_feedforward_layernorm`. Expert weights arrive stacked as one
`[experts, ...]` parameter; `plane_bind_part` slices an expert out as a view
rather than copying it. The mixture branch stays one lane wide, because each
token picks its own experts.

`session_prime` consumes every prompt token but the last in batches of
`KERN_LANE_LIMIT`, skipping the output head for each, because those logits
are never read. Batching turns each projection into a matrix product: a group
of packed codes is unpacked once and reused by every lane. The cache write
and the attention read stay in position order inside the batch, because a
sliding layer's ring is narrower than the batch. The caller feeds
the last token to `session_step`, which is the only call that pays for the
head.

`session_pick` draws a token: repetition penalty over a recent window,
temperature, top-k, top-p, then a draw from an xorshift stream. A
temperature of zero short-circuits to the maximum.

## 4. Data flow

```
folder ──► config.json ──► model_form ──► layer_wing[]
       │
       ├─► model.safetensors ──► store_span[] ──► plane[]
       │                          (mmap, zero copy)
       │
       └─► tokenizer.json ──► token_book

text ──► token_encode ──► ids ──► session_prime ──► session_step ──► logits
                                       │                 │
                                  key/value cache   session_pick ──► id ──► token_decode ──► text
```

## 5. Formats read

| file                      | needed for                                     |
| ------------------------- | ---------------------------------------------- |
| `config.json`             | architecture, layer types, rotary, quantization |
| `generation_config.json`  | default sampling and stop ids, when present     |
| `model.safetensors`       | the weights                                     |
| `model.safetensors.index.json` | shard map, when the checkpoint is split   |
| `tokenizer.json`          | vocabulary, merges, special tokens              |

## 6. Testing

`app_test.c` covers every layer that can be checked without the checkpoint:
the path and slicing helpers, the thread pool, the JSON reader including
escapes and rejection of malformed input, a hand-built safetensors file, the
half-precision conversions, `pack_read` against an independent reference bit
writer, the zero-point layout, `plane_row` against a matrix whose dense form
is known, the quantized matrix product against that same dense form, RMSNorm,
GELU, softmax stability and shift invariance, the rotary transform and its
inverse, both rotary schedules, and a tokenizer round trip over a small
synthetic vocabulary. `test_expert` checks that an expert slice of a stacked
parameter lands on the right rows, and `test_wing` writes a complete
miniature checkpoint — config, tokenizer, and every tensor the loader binds,
once dense and once with a mixture block — and asserts that a batched prefill
reaches exactly the logits produced by feeding the same tokens one at a time.

`test_gemma` writes a safetensors fixture in the shipped export's layout — a
four bit plane, a signed eight bit plane, and a two bit embedding table with
two groups to a row — and checks the decode of each, the group size the scale
shape implies, the bit width the column hint implies, the two activation
steps, the refusal of a column hint no packing explains, and the eight bit dot
product, which reads its bytes directly and so has to take the sign flip that
the row decode takes.

`app_test.py` drives the built binary and the transformers reference over
the same prompts. It compares the token ids, the rank-one token, the top-k
overlap, the largest absolute logit gap, and the greedy continuation, then
reports the decode throughput of each side and their ratio. The last two are
measured rather than assumed: the reference is run a second time with its
tokens fed one at a time behind its cache, which is the same arithmetic in
another summation order, and the engine is held to twice what that moves. On
a checkpoint that rounds its activations onto a static grid it moves by whole
units, and a fixed tolerance would report the checkpoint's own rounding as an
engine fault. It skips cleanly, exiting zero, when the checkpoint or the
python packages are absent, so it is safe in a pipeline.

### 6.1 The synthetic oracle

`app_test.py` needs the shipped checkpoint. `app_fake.py` and `app_diff.py`
need nothing but the python packages, and together they answer the question
that matters — does this engine compute the same numbers as the reference —
without any download at all. `python3 run.py parity` runs the whole thing.

**`app_fake.py` builds the checkpoint.** It constructs a `Gemma4Config` at a
size that fits in a test — 32 hidden, 4 layers, 4 heads — instantiates the
reference model, reseeds every parameter, and calls `save_pretrained`. The
reference library writes `config.json` and `model.safetensors` itself, so the
tensor names, shapes, and layouts come from upstream rather than from a
reading of upstream. That is the difference between this and `test_wing`,
which is hand-built and therefore agrees with whatever the loader believes.

Reseeding is not cosmetic. The reference initializer sets every norm weight
to exactly 1.0, which hides any bug in how a norm is indexed, and leaves
whole families of tensors identical, which hides any bug in which one is
picked. `form_seed` gives every parameter its own values, and `form_check`
refuses to emit a model with a parameter that stayed near zero — a muted
branch agrees with any implementation at all.

A second mode packs the float checkpoint the way the QAT export is packed:
symmetric group-wise scales, a bit-packed `weight_packed` as I32 built by
`compressed_tensors` itself, plus `weight_scale` and `weight_shape`. It then
dequantizes in python and writes that as a third checkpoint. The engine
reading the packed files is compared against the reference reading the
dequantized ones, which isolates the unpacking and the scale convention from
everything else.

**`app_diff.py` compares.** Final logits tell you that something is wrong,
never where. A build with `-DAPP_TRACE` — `run.py build --trace` — writes a
named activation snapshot to the file named by `IGLLM_TRACE`: the embedding,
then each layer's attention output, mixture or feed-forward output, and
residual, then the final norm and the logits. The reference side captures the
same tensors with forward hooks. The harness walks them in order and stops at
the first one that exceeds tolerance, which names the broken function.

The tolerance is measured, not guessed. `reference_floor` runs the reference
twice over the same tokens — once over the whole sequence, once one token at
a time behind its cache — which is the same arithmetic in a different
summation order, and takes the largest gap as the noise floor. Tolerance is
eight times that. The dump is compiled out by default, so a normal build
carries none of it.

The sweep covers the axes that change code paths: dense and mixture blocks,
sliding and full attention, every packed bit width from two to eight, group
sizes that do and do not divide the row, the double-wide feed-forward, shared
key and value projections, and prompt lengths that straddle both the sliding
window and the prefill chunk.

The oracle itself was checked by breaking the engine on purpose: perturbing
one expert weight makes it report that layer and stay silent about the ones
before it. An oracle that never fails has not been shown to work.

**`app_diff.py --model <folder>` compares against a real checkpoint**, and
asks something different of it, because the shipped export rounds every
activation onto a static grid. A sum that lands on a half step falls one way
in single precision and the other way in double, so a last-bit difference
becomes a whole step and then compounds, and the reference does not reproduce
itself: the same graph in double precision agrees with single to a part in ten
million through the first layers, moves by whole steps from the middle of the
stack, and by whole units in the logits. There is no single-precision answer
to agree with. So the harness reports how deep each side stays exact, requires
the embedding and the first layer — which nothing can have rounded twice yet —
to agree outright, and holds the end of the stack to how far the reference
moves there. That last judgement is made over the whole run rather than prompt
by prompt: whether a given perturbation lands on a half step is a lottery, and
a prompt where the reference happens to flip nothing says nothing about what
the engine may do.

## 7. Extending

**A new backend.** Write a `back_open` variant that fills `back_desk` with
device entry points, and add a device handle beside `plane`. Nothing in the
model or session layer changes.

**A new quantization format.** Add a `plane_form` value and a case in
`plane_bind`, `plane_row`, and `kern_row_code`. The rest of the engine sees
only `plane`.

**A new architecture.** Extend `model_form` and `layer_wing`, and add the
matching branch in `session_layer`. The loader, tokenizer, kernels, and
sampler are all architecture-neutral.
