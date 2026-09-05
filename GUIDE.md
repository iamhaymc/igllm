# GUIDE

A complete tour of the igllm implementation: what each file holds, how a
prompt becomes a token, which formats are read, how the kernels are built,
and where an accelerator would attach.

---

## 1. File map

| file          | role                                                         |
| ------------- | ------------------------------------------------------------ |
| `app_core.c`  | the engine: a public interface followed by eleven layers     |
| `app_main.c`  | the command line front end                                   |
| `app_test.c`  | the unit tests                                               |
| `app_test.py` | parity and throughput comparison against transformers        |
| `app_fake.py` | builds a synthetic checkpoint and quantizes it               |
| `app_diff.py` | layer by layer comparison against transformers, and the seam |
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
- `app_media` — the embedding rows one picture or one clip turned into.
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

- `kern_dot_real` — dot product against an `f32`, `f16`, or `bf16` row. The
  `f32` case has a vector path on all three targets.
- `kern_dot_code` — dot product against a packed row, specialized for the
  two, four, and eight bit cases and general otherwise. All three have vector
  paths on all three targets: a nibble unpacks with whole-vector shifts and
  masks, a quarter byte with those or with the table below, and a whole byte
  needs only the flip. The two narrow widths require the group to start on a
  byte boundary and fall back to the bit-stream loop when it does not.

  Every vector path carries two accumulators rather than one. The arithmetic is
  cheap enough that a single chain waits on the latency of its own add rather
  than on the work: splitting it took a four bit row 1.6 times quicker on SSE2
  and 1.4 on AVX2, for nothing but a second register.

  At two bits the unpacking is read out of `kern_code_two` instead, a four
  kilobyte table of four floats indexed by the byte that holds them, built by
  the preprocessor so it costs no startup and lives in read-only memory. A byte
  is exactly four codes at that width, so a shift, a mask and a convert per code
  become one sixteen byte read, and the values are the codes themselves, in the
  lanes the unpacking put them in — the two paths agree to the last bit rather
  than to a tolerance. On the export's widest row it takes the fused dot from
  0.0039 s to 0.0023 s on SSE2.

  The wide path keeps its shifts, and the reason is worth recording because the
  table looks like it should win everywhere. Filling one 256 bit vector from the
  table costs two narrow loads and an insert, against one broadcast, one
  variable shift and one mask, and it measures slower: 0.0018 s against 0.0016 s
  on the same row.
- `kern_row_code` — one output row of a quantized matrix, formulated as

  ```
  row = Σ_groups gain * ( Σ_j a[j]*code[j]  -  (offset+zero) * Σ_j a[j] )
  ```

  The per-group activation sums are computed once per matrix product by
  `kern_group_sum` and shared by every row. This removes one subtraction
  per element from the inner loop and accumulates in a wider range.
- `kern_code_spread` — a run of codes unpacked into floats, with the same
  vector paths `kern_dot_code` fuses into its own loop. It exists because a
  batch has someone to share the decode with and a single vector does not.
  Written out rather than summed, a byte of two bit codes *is* a row of
  `kern_code_two`, so that width is a sixteen byte copy: 0.0031 s against
  0.0023 s on SSE2, and 0.0092 s against 0.0023 s in the plain loop the
  remainder and a host without a vector path both run.
- `kern_row_code_many` — the same row against several activation vectors at
  once. A group of codes is spread into a small float scratch and dotted
  against every lane, so a batch pays the decode cost of a single vector. When
  that spread was a scalar walk of the bit stream, a sixteen lane batch cost
  three and a half times what the same sixteen lanes cost one at a time, and
  batching lost to the thing it exists to beat.
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

### 3.8 Media layer

Everything a picture or a sound passes through before a tower sees it. The
layer knows nothing about models: it turns bytes on disk into a rectangle of
floats or a run of samples.

- `flat_grid` — a band-interleaved raster. A mel spectrogram is one of these
  too, with one band, the frame as the row and the filter as the column, which
  is what lets the tower layer treat a picture and a sound the same way.
- `puff_run` — an inflate reader: stored, fixed and dynamic Huffman blocks,
  with the zlib wrapper skipped when one is present. It is here because a PNG
  cannot be read without one and a third party decoder is not an option.
- `png_read` — eight and sixteen bit grey, RGB, palette, and the two alpha
  forms, non-interlaced. All five line filters are undone. Alpha is dropped
  rather than composited: inventing a background is a preprocessing choice
  this layer has no business making.
- `pnm_read`, `bmp_read` — binary `P5`/`P6`, and uncompressed 24 or 32 bit
  bitmaps. `image_read` picks between the three from the leading bytes rather
  than the file name.
- `grid_scale` — a separable bicubic, the `a = -0.5` member of the family.
  The support widens with the reduction factor, so shrinking an image averages
  over everything it passes rather than sampling sixteen pixels out of a
  thousand; centres are half-pixel, so the two images cover the same area.
  `grid_bands` folds three bands onto one by the luma weights, or spreads one
  over three.
- `wave_read` — RIFF wave: 8, 16, 24 and 32 bit integer and 32 bit float,
  `WAVE_FORMAT_EXTENSIBLE` included, with channels averaged rather than
  dropped. `wave_rate` resamples linearly, which is below what a filterbank
  can see.
- `wave_spin` — an in-place radix-2 transform. `mel_bank` builds triangular
  filters spaced evenly on the HTK mel scale at fractional bin positions, and
  `mel_make` produces the log mel spectrogram: a periodic Hann window, a stated
  transform length, a floor under the logarithm so a silent band cannot reach
  negative infinity, and a frame count that stops where the audio does.

### 3.9 Model layer

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

### 3.10 The vision and audio towers

Two encoders sit beside the text stack, each ending in a projection into the
text hidden width. They are bound only when `config.json` carries a
`vision_config` or an `audio_config` **and** the weights carry the matching
tower; a configuration that describes one the export left out is a text-only
checkpoint, not a broken one, and loads as such.

`tower_prefix_pick` finds the names the exporter used, probing both shapes the
reference writes — vision layers under an `encoder` module, audio layers
directly under the tower — and `tower_bind_line` looks one level deeper than the
name of a projection suggests, because the reference wraps every tower linear in
a module that can clip its input and its output. As everywhere else in the
loader, the weights decide the shape: the hidden width comes from `q_proj`'s
column count, the head size from its row count over the configured head count,
the feed-forward width from `gate_proj` or `ffw_layer_1`. Only what a tensor
cannot record is read from the configuration — the patch size, the pooling
factor, the analysis window, the attention span.

Both towers' projections go through `plane_lift_many`, the same function the
text stack's projections go through, so the checkpoint's packed weights and its
calibrated activation ranges are handled without a second code path. The shipped
export quantizes the vision tower to eight bits and the audio tower to two, and
`plane_bind_gemma` recovers each width from the stored shape rather than from
the configuration's regular expressions.

**The vision tower** is the text layer with the causal mask taken off:
`input_layernorm` → attention → `post_attention_layernorm` → residual, then
`pre_feedforward_layernorm` → gated MLP → `post_feedforward_layernorm` →
residual. That is `session_layer` without the per-layer embedding gate and
without the mixture branch, which is why `tower_feed` is one function.

It takes a **variable resolution** rather than a fixed square. `vision_grid_pick`
preserves the aspect ratio, caps the area by the patch budget the soft-token
limit implies, and rounds both sides down to a multiple of `pool_size *
patch_size` so the pooler's windows divide exactly. `vision_patch_cut` then cuts
the picture into patches; inside one patch the samples run row, then column,
then band — the band is the fastest axis, which is what the reference's flatten
produces. Pixels arrive in `[0,1]` and are scaled to `[-1,1]`, a step the
reference does in the model rather than in the preprocessor.

Positions are two dimensional and enter twice. A learned table holds one row per
column index and one per row index — `position_embedding_table` is the two
stacked — and the patch's two rows are summed into its embedding. Then
`rope_grid_wave` and `rope_grid_turn` give the first half of every head the
patch's column and the second half its row, **rotating each half within itself**:
channel *j* of a half pairs with channel *j + half/2* of the same half. That is
not the text stack's rotate-half, which pairs across the whole head, so the
vision tower has its own turn. It needs a head size divisible by four, and a
tower whose head does not divide is refused rather than guessed at. Attention is
bidirectional and scaled by **one**: the query and key norms absorb the usual
`1/sqrt(d)`, exactly as they do in the text stack. Values carry a norm without a
scale.

After the stack, `vision_pool` averages over `pool_size` squared windows and
scales by the square root of the hidden width.

That attention is the whole cost of an image: at the shipped export's patch
budget a picture is two thousand three hundred and forty patches scored against
themselves, sixteen layers over. It is also the one loop in either tower that
divides cleanly, so it is the one that goes to the pool — and it goes a head at
a time, because a head's keys are sixty-four floats in every seven hundred and
sixty-eight where they lie. Gathering one head into a run of its own turns a
seven megabyte walk per query into six hundred kilobytes that stays in cache.
The two together took an image from four minutes thirty-seven to one minute
fifty-two on four cores, without moving a single number.

**The audio tower is a conformer**, not a transformer, and the difference is
worth stating because the two look alike from a distance. A layer is

```
feed-forward (folded back at half weight)
norm, chunked local attention, norm, residual
a gated depthwise convolution over time
feed-forward (folded back at half weight)
norm
```

so it carries two feed-forwards rather than one and a convolution module the
text stack has no equivalent of. `sound_wing` and `sound_bind` are separate from
the vision tower's for that reason.

The clip is read, resampled to the tower's rate, and turned into a log mel
spectrogram whose every convention comes from the checkpoint's
`preprocessor_config.json`: a periodic Hann window, a stated transform length
rather than a derived one, half a window of silence prepended so the first frame
is centred on the first sample, a frame cut one sample longer than the window
before the last sample is dropped, a **magnitude** spectrum rather than a power
one, and a floor added under the logarithm rather than a clamp inside it.
Squaring instead of taking the magnitude is the easy mistake here: it survives
every sanity check and only a comparison against the reference finds it.

What the engine does **not** do is pad the clip out to a whole block of a
hundred and twenty-eight samples, which the extractor does before it frames
anything. That is deliberate, and it is worth saying why, because the extractor
plainly hands those extra frames back and matching its output shape looks like
the faithful thing to do.

The extra frames are masked. `input_features_mask` marks them, the subsampler
zeros them between its two convolution stages, the conformer's attention is
built to exclude them, and `Gemma4Model` keeps only the rows the tower's output
mask leaves — `audio_features[audio_mask_from_encoder]`. So they reach nothing.
What they would change, if the engine counted them, is the number of rows it
thinks the clip is worth, because a row survives the two stride-two stages only
if the frame four times its index is real. Counting the padding puts a soft
token there that the reference never asks for: four thousand samples reach
twenty-four frames and pad to twenty-five, and twenty-four frames are six soft
tokens where twenty-five would be seven.

Measured against the extractor over clip lengths from a twentieth of a second to
ten seconds, the frame count the engine computes is the live count the mask
marks on every one of them, and `ceil(live / 4)` is what the processor's own
`_get_num_multimodal_tokens` asks for on every one of them. The padding is a
detail of how the extractor batches, not of what the model reads.

What the engine does do is stop where the processor stops. The framing above is
the feature extractor's; the **budget** is the processor's, and it is written in
a third file beside the other two, `processor_config.json`: `audio_seq_length`
soft tokens of `audio_ms_per_token` milliseconds each, 750 and 40 on the shipped
export, which is half a minute of audio. A clip inside that is worth its own
live frames on both sides, which is why the two agreed for as long as every clip
tested was a short one. A longer clip the processor pads or trims to fit, and
the trim is the half that shows: past the ceiling the reference stops reading,
and the engine left to itself would keep going and hand the prompt soft tokens
the reference never asked for — a thirty-five second clip is 875 rows framed and
750 rows budgeted.

`config_budget_read` reads the pair, `sound_budget_samples` turns it into a
sample count, and `media_audio` applies that to the resampled clip *before* a
frame is taken from it. Cutting the samples rather than capping the rows is the
whole point: a cap would agree on the count and disagree on the last row, whose
frames would have been drawn from audio the reference never read. The two halves
of the processor's configuration agree that this is the right place to cut —
750 tokens of 40 milliseconds is 480 000 samples, which frames to 2999 live
frames, which is `ceil(2999 / 4) = 750` rows, the budget exactly and not a row
over. The suite holds a clip four times the budget to the clip that ends at it,
row for row rather than merely the same length, and `app_media` carries a
`cut_flag` so the command line can say the tail was dropped rather than answer
about half a clip without saying so.

`sound_stage` then runs the subsampler — two convolutions over (frame, filter),
each followed by a mean-subtracting `kern_norm_layer` across the channels it
produced and a rectifier — and the map is folded into the hidden width with the
filter axis outside the channel axis, which is the order the reference's permute
leaves behind.

`sound_attend` writes the attention as the causal window that the reference's
chunking and its sliding mask agree on. The reference cuts the sequence into
chunks, gives each a context window, and masks that window down with a rule of
`(attention_context_left - 1, attention_context_right)` — and that rule is a
**strict** inequality, so a query reaches `attention_context_left - 1` keys at or
before itself. Getting that bound wrong by one is invisible in every shape and
changes every number; it is the one thing in this tower that a shape check
cannot catch. What survives both the chunking and the mask is exactly the
sliding window, so the block machinery is an efficiency device rather than part
of the arithmetic, and the relative shift it needs collapses into indexing the
position row by the lag. The score is a content term plus a term against a
projection of the sinusoid of that lag, the query is scaled per head channel
through a softplus and the key by a constant, and the sum is softcapped with a
tanh before the softmax.

**The projector.** Each tower ends in one norm and one linear into the text
hidden width. The norm carries no scale in the reference, so a missing weight
means "normalize without one" rather than "do not normalize". The audio tower
widens through its own `output_proj` first. `plane_bind_turn` accepts the
projection stored either way round, because a multi-modal projection is
conventionally `[in, out]` while every other weight in the checkpoint is
`[out, in]`.

**Where the rows go.** `media_image` and `media_audio` return an `app_media` —
one embedding row per placeholder token. The caller lays those rows against the
ids and hands both to `session_prime_media`, which substitutes a row for the
embedding lookup of the lanes it is given. The substitution happens *after* the
token path's `sqrt(hidden_size)` scale, not before it, so a projector's output
is already in the units the residual stream carries.

What the id at a filled lane is used for afterwards is the per-layer embedding,
and there the reference does not read the placeholder: it rewrites every media
position to the pad id before it computes the token-identity half of the
per-layer input. So `session_pass` seeds that half from `pad_id` at a filled
lane, and nothing of the placeholder reaches the stack — the id under a filled
lane cannot change an answer, which is what `test_tower` asserts.

**The ids around the run.** A run of placeholders is not laid down bare. The
processor opens and closes each one — `boi_token_id` and `eoi_token_id` for a
picture, `boa_token_id` and `eoa_token_id` for a clip — and the model reads
those two ids as ordinary text, so a prompt built without them is a prompt the
reference never sees. `token_media_run` writes one span's ids and reports where
its placeholders begin, and `token_frame_media` splices that between the role
marker and the user's words, which is where a multi-modal template puts it. The
caller lays its rows against the reported index rather than against the opener.

An export that records neither bracket is read as having none; half a bracket is
treated as none at all, because half of one is a prompt nothing lays down.

A tower allocates its own scratch per call rather than per session: it runs once
for a prompt and never inside the token loop.

### 3.11 Token layer

`token_load` reads `tokenizer.json`: the vocabulary, the merge list, the
added tokens with their special marks, and the metaspace behaviour from the
normalizer and pre-tokenizer records. Byte-fallback entries of the form
`<0xNN>` are indexed into a 256 entry table.

`token_split` maps text to UTF-8 runes, replacing spaces with the metaspace
mark and prepending the mark when the tokenizer asks and the caller says this
text begins a chunk. The mark belongs to the start of a chunk rather than to the
start of a call: the reference splits its input on the special tokens and
normalizes each chunk on its own, so a piece that follows a special id is marked
and a piece continuing the line before it is not. That is why `token_frame_media`
marks `user\n` and the words after a media run, and does not mark the words when
nothing was attached.

`token_encode_book` then applies the standard byte-pair merge loop, always
taking the lowest ranked adjacent pair, and falls back to byte tokens for
anything the vocabulary does not hold. `token_decode_book` reverses that, turning the
metaspace mark back into a space, expanding byte tokens, and emitting
nothing for control tokens.

`token_frame` applies the Gemma chat frame for instruction-tuned prompts:
the sequence marker, then a user turn, then the opening of a model turn.
Gemma 4 writes the turn markers `<|turn>` and `<turn|>`, and the older
`<start_of_turn>` and `<end_of_turn>` are probed behind them, since nothing
else in a checkpoint distinguishes the two vocabularies. `chat` and `logits`
both frame; `--raw` and every other task do not.

### 3.12 Session layer

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
folder ──► config.json ──► model_form ──► layer_wing[], tower_gear[]
       │
       ├─► model.safetensors ──► store_span[] ──► plane[]
       │                          (mmap, zero copy)
       │
       └─► tokenizer.json ──► token_book

text  ──► token_encode ──► ids ───┐
                                  ├─► session_prime_media ──► session_step ──► logits
image ──► image_read ──► resize ──┤          │                     │
              └► vision_run ──► rows         │                session_pick ──► id ──► text
sound ──► wave_read ──► mel ──────┤     key/value cache
              └► audio_run  ──► rows
```

An image or a clip enters as a bracketed run of placeholder ids with one
embedding row laid against each placeholder; everything after that is the text
path.

## 5. Formats read

| file                      | needed for                                     |
| ------------------------- | ---------------------------------------------- |
| `config.json`             | architecture, layer types, rotary, quantization, the tower blocks |
| `generation_config.json`  | default sampling and stop ids, when present     |
| `model.safetensors`       | the weights                                     |
| `model.safetensors.index.json` | shard map, when the checkpoint is split   |
| `tokenizer.json`          | vocabulary, merges, special tokens              |
| `preprocessor_config.json` | the audio analysis window, when present        |
| `processor_config.json`   | the clip's soft token budget, when present      |
| `.png`, `.pnm`, `.bmp`    | a picture for the vision tower                  |
| `.wav`                    | a clip for the audio tower                      |

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

`test_puff`, `test_image`, `test_scale`, `test_wave` and `test_mel` cover the
media layer. The inflate reader is held against a stored block assembled in the
test and against a dynamic Huffman block produced by an independent compressor
over four hundred bytes the test can regenerate from a formula; a truncated
stream and one that would overrun its room are both required to be refused. The
png fixture is a twelve by eight picture written by an independent encoder with
the five line filters used in turn, and the same picture is written again as a
pnm and as a bottom-up bitmap, so the three readers are held to one answer. The
resize is checked four ways: a constant survives in both directions, a ramp
stays a straight line where the taps fit, the separable implementation matches a
direct two dimensional gather, and folding three bands onto one takes the luma
weights. The wave reader has to step over an unknown chunk, average its channels
rather than drop one, and halve its sample count when the rate halves. The fast
transform is held against the discrete one it is a fast way of computing, the
filters against the shape the mel scale implies, and a pure tone against the
filter that covers it.

`test_tower` writes a miniature multi-modal checkpoint in the shipped export's
arrangement — the reference's module names, its wrapper around every projection,
its two axis position tables, its conformer — and runs the vision tower against
a definition of it written out separately in the test file. The two sides share
the weights and the decoded picture and nothing else, so agreement says the
arrangement is right rather than merely self-consistent.

The audio tower is not reproduced a second time in C. It is a conformer with a
dozen interacting parts and a second transcription here would mostly be a copy
of the first; what it gets instead is a check on each piece peculiar to it — the
mean-subtracting norm, the silu, the softplus — plus the end-to-end run, the
causal property that a longer clip repeats a shorter one's first row, and the
requirement that a clip which sounds different reaches a different answer. The
tower as a whole is held against the real reference by `app_diff.py --media`,
which is a stronger test than anything written here could be.

Both cases then check that a substituted embedding actually reaches the stack:
the same ids primed with and without the tower's rows have to land on different
logits.

`test_gemma` writes a safetensors fixture in the shipped export's layout — a
four bit plane, a signed eight bit plane, and a two bit embedding table with
two groups to a row — and checks the decode of each, the group size the scale
shape implies, the bit width the column hint implies, the two activation
steps, the refusal of a column hint no packing explains, and the eight bit dot
product, which reads its bytes directly and so has to take the sign flip that
the row decode takes.

One section is the exception, and it is the last one. `test_shot` wants the
checkpoint: it looks in `IGLLM_MODEL`, then beside the tree, and when it finds
the shipped export it runs three prompts through the whole stack — frame,
prefill, one step — and holds each to numbers recorded from the build the seam
comparison judged against the reference. The ids of the frame have to be exact,
the leading id has to be exact, the top eight logits have to come back within
2.0, and none of those eight may fall out of the top sixteen. The slack is
measured rather than guessed: between this host's SSE2 and AVX2 builds, over
five prompts, rank one never moved, no id in a top eight fell below rank eleven
in the other build, and the logits moved by at most 1.30. The bar sits above
that and well inside the 2.30 to 5.33 the reference moves against itself when
its own input is nudged by a millionth, so what it catches is a layer wired
wrong rather than a last-bit difference. What it buys is that the parity result
outlives the host it was measured on: a tree that has the checkpoint but no
python still fails if the stack stops reaching the same answer. It costs about
twenty seconds. A folder holding some other checkpoint is passed over rather
than failed, and `IGLLM_MODEL=none` turns the section off.

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

**`app_diff.py --media` walks the two towers against the reference itself.**
Each tower is built from the reference's own module and loaded with the
checkpoint's own weights, its packed tensors decoded by the reference's own
`QuantizedLinear`. A tower is a couple of hundred megabytes and the reference
gives each one a `base_model_prefix`, so it can be built alone rather than
standing the whole model up to look at one encoder — quicker, lighter, and it
keeps a tower's arithmetic isolated from everything around it. So the towers are
held to the same standard as the text stack — upstream, on the real weights —
rather than to a second reading of the same description.

Both sides start from the rows the engine says it read — the normalized patches,
or the mel frames — because the png reader, the resize and the filterbank each
have an independent definition in `app_test.c` already, and what is in question
here is the tower above them. The filterbank is separately held against the
reference's own `Gemma4AudioFeatureExtractor`.

The bar is measured rather than picked. On a checkpoint that rounds every
activation onto a static grid there is no single answer to agree with, so the
reference's own input is nudged by a part in a million and how far that carries
is the size of a difference that means nothing. On the shipped export the vision
tower's rows differ by 2.08 where that nudge moves the reference by 2.62, and
the audio tower's by 2.38 where the nudge moves it by 5.36 — in both cases the
engine is closer to the reference than the reference is to itself under a change
that should not matter. On synthetic float weights, where there is no grid, the
vision tower agrees to 1.2e-07 and the audio tower to 2.2e-05.

**`app_diff.py --seam` walks the join between them.** A tower that agrees with
the reference and a text stack that agrees with the reference still say nothing
about where they meet: the ids the processor lays down around a run of soft
tokens, and what the model makes of them, exist only when the whole graph runs.

It is two halves, because they can be reached separately.

The *layout* half asks the reference's own `Gemma4Processor` where the
placeholders go. A processor is a tokenizer and three small configurations, so
this needs no weights at all and runs against the shipped export as easily as
against a synthetic one. The chat template frames the turn, `replace_image_token`
and `replace_audio_token` substitute a bracketed run for the marker it wrote, and
the ids that come out have to be the engine's, id for id. Two counts are checked
beside them: what the processor's own arithmetic says a picture of that size is
worth, which is the aspect-preserving resize against the patch budget, and what
it says the clip is worth. The second is the feature extractor's framing, which
was an open question and is now settled — the engine's frame count is the live
count `input_features_mask` marks, on every clip length it has been measured
against — so it is checked like the rest. Above `audio_seq_length` tokens the
answer is the budget rather than the framing, on both sides; no clip that long
has been put through this harness, because the reference's forward on one is a
larger thing than the count it is being asked for.

The *graph* half runs `Gemma4ForConditionalGeneration` over those same ids and
compares the distribution. That wants the whole model in memory, which the
shipped export does give: its weights stay packed and are decoded per forward,
so it loads in about two and a third gigabytes. `app_fake.whole_build` writes a
synthetic one as well — the reference's own model class, both towers, both
projectors, a tokenizer carrying the media tokens, and the processor files
beside them — which runs in seconds where the real one takes minutes. The towers
are fed the rows the engine says it read, as `--media` feeds them, so what is
measured is the seam rather than the png reader or the filterbank.

The bar is the reference's own movement, and how that is measured depends on
what was attached. With a picture or a clip the input is nudged by a millionth
and what reaches the logits is the answer. With nothing attached there is
nothing to nudge, and it is tempting to call the floor zero because the ids are
exact on both sides — but that does not follow on a checkpoint that rounds every
activation onto a static grid, where a sum landing on a half step falls one way
here and the other way there. So the same tokens are fed again in two chunks
behind the cache, which is the same arithmetic in another summation order and
the measure `reference_floor` already uses on the text stack. A floor of zero
was not a measurement but the absence of one, and it called the shipped
checkpoint wrong until it was replaced.

A forward on a quantized checkpoint costs about a gigabyte and a half above the
weights, because reading one row of an embedding dequantizes the whole table,
and the case with both a picture and a clip attached wants two of them. Held in
one process across four cases that does not fit, and — worse than not fitting —
which case fits depends on what ran before it: the same case passed alone and
skipped after `--media` had run in the same shell.

So each case is judged in a process of its own. `--seam` spawns itself with
`--seam-case`, and the child loads the model, runs the graph half and exits,
handing its memory back. The engine's work is not repeated: the child is given
the path to the trace the parent already wrote and reads it directly, so the
cost is one model load per case rather than one engine run. The parent releases
that trace before the child starts, because one picture leaves half a gigabyte
of rows behind and the parent needs three numbers out of it.

The child's exit code is the report, and it is deliberately not 1 for a
disagreement: 1 is what python returns for an unhandled traceback, and a parent
that cannot tell the two apart counts a crashed case as a fault and prints
nothing to say so. It answers 0 or 2. Anything else is a child that died, and
the parent prints its last lines and sorts it into the host running out of
memory, which is a skip, or anything else, which is the harness's own fault and
is labelled as such.

Three engine faults came out of writing it and running it, none of which any
single-tower comparison could have reached: the run was laid down without the
ids the processor brackets it with, the per-layer embedding at a filled lane
read the placeholder where the reference reads the pad id, and the floor that
was supposed to judge all of it measured nothing at all.

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

**A new modality.** Add a reader to the media layer, a `tower_form` and a
`tower_gear` beside the two that exist, and one `*_run` that fills embedding
rows. `tower_feed` already carries the residual arrangement; what a new tower
has to supply is how a position enters the score and how its input becomes a
row. Nothing in the session or the sampler changes: a tower's output is
substituted for a placeholder token's embedding and the text path takes it from
there.
