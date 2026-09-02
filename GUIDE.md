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
| `run.py`      | install, build, test, check, run, clean workflows            |

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

Gemma 4 E2B IT QAT ships in the compressed-tensors **pack-quantized**
format. A quantized linear is four tensors:

| tensor              | meaning                                          |
| ------------------- | ------------------------------------------------ |
| `weight_packed`     | `int32` words holding a dense bit stream of codes |
| `weight_scale`      | one scale per output row and input group          |
| `weight_zero_point` | optional, packed along the row axis               |
| `weight_shape`      | the logical `[out, in]` shape                     |

The packing is dense and little-endian: element *i* of a row occupies bits
`[i*bits, i*bits + bits)`, and rows are padded up to a 32-bit boundary. On
a little-endian host the mapped bytes are therefore usable **as they are** —
`pack_read` walks the bit stream directly with no repacking pass and no
second copy of the weights. Codes are stored with a sign offset folded in,
so the dequantized value is

```
w = (code - offset - zero_point) * scale
```

`plane` is the storage record for any weight. It is either `PLANE_REAL` —
a plain `f32`, `f16`, or `bf16` matrix read in place — or `PLANE_CODE` — the
packed form above. `plane_row` decodes a whole row when a row is what is
wanted, for example a single embedding lookup. `plane_bits_of` recovers the
bit width from the shape and the packed word count, so a checkpoint that
mixes widths across tensors, as this one does, is handled without trusting
the configuration.

`quant_rule` and `quant_book` carry what `quantization_config` declares:
bit width, group size, symmetry, strategy, and the ignore list. When the
configuration declares quantized input activations, `quant_act` reproduces
the reference fake-quantization per token before the matrix product.

### 3.6 Kernel layer

Every kernel has a plain scalar definition and, where it pays, a vector
path chosen at compile time behind one macro layer: `APP_SIMD_AVX2`,
`APP_SIMD_SSE2`, `APP_SIMD_NEON`, or none. `back_flavor` names the path that
was selected.

- `kern_dot_real` — dot product against an `f32`, `f16`, or `bf16` row.
- `kern_dot_code` — dot product against a packed row, specialized for the
  two, four, and eight bit cases and general otherwise.
- `kern_row_code` — one output row of a quantized matrix, formulated as

  ```
  row = Σ_groups gain * ( Σ_j a[j]*code[j]  -  (offset+zero) * Σ_j a[j] )
  ```

  The per-group activation sums are computed once per matrix product by
  `kern_group_sum` and shared by every row. This removes one subtraction
  per element from the inner loop and accumulates in a wider range.
- `kern_mat_vec_band` — one band of rows, the unit of work given to the pool.
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
  void (*norm_rms)(struct back_desk *, const float *, const float *, int, float, float *);
  void (*soft_max)(struct back_desk *, float *, int);
  void (*gelu_gate)(struct back_desk *, float *, const float *, int);
  void (*rope_turn)(struct back_desk *, float *, int, const float *, const float *);
} back_desk;
```

`back_open` binds the CPU implementation. The model and session layers never
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
the key-value head count, from the row count of `k_proj`. A checkpoint whose
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
different transformers versions load without a flag.

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

### 3.10 Session layer

A session owns the caches and the scratch. `session_open` allocates once,
sized from the model, and nothing in the token loop allocates again.

Key and value caches are sized per layer. A sliding layer needs only its
window and uses a ring, indexed by `position % cache_span`. A full-attention
layer, and any layer that is the source for a sharing layer, keeps the full
context length. Layers that share simply read the source layer's cache and
never write their own. That sizing is what keeps the cache in the tens of
megabytes rather than the better part of a gigabyte.

`session_pass` runs one token:

1. embedding lookup, scaled by `sqrt(hidden_size)`
2. the per-layer embedding block: table lookup scaled by `sqrt(ple_size)`,
   the model projection scaled by `hidden_size^-0.5`, normalized, and mixed
   with the token identity at `2^-0.5`
3. for each layer, `session_layer`:
   - `enter_norm` → attention → `after_attn_norm` → residual add
   - `before_feed_norm` → gated MLP → `after_feed_norm` → residual add
   - the per-layer embedding gate → `after_ple_norm` → residual add
   - a scalar layer gain
4. the final norm, the output head, and optional logit softcapping

`session_attend` is the attention itself. Queries and keys are normalized
with their own RMSNorm weights, rotated, and scored against the cache.
Attention scaling is `1.0`, not `1/sqrt(d)`; the query and key norms absorb
it, which is what the reference does. Values carry a norm without a scale.
When `attention_k_eq_v` is set, a full-attention layer has no `v_proj` and
uses the raw pre-norm key projection as its values.

`session_prime` consumes every prompt token but the last, skipping the
output head for each, because those logits are never read. The caller feeds
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
synthetic vocabulary.

`app_test.py` drives the built binary and the transformers reference over
the same prompts. It compares the token ids, the rank-one token, the top-k
overlap, the largest absolute logit gap, and the greedy continuation, then
reports the decode throughput of each side and their ratio. It skips
cleanly, exiting zero, when the checkpoint or the python packages are
absent, so it is safe in a pipeline.

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
