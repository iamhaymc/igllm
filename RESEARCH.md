# Beyond the TODO: CPU text and vision acceleration

Research snapshot: 2026-09-06, revised 2026-09-07 after 0.8.9 and after reading
`ggml-org/llama.cpp` at commit `465e49b`. This is an experiment shortlist, not a
performance claim.

**What this file is now.** It opened as twelve ideas written on the assumption
that `TODO.md` was finished. Seven of them are no longer research. One is built.
Six are ordinary engineering with a known payoff, and they have moved to
`TODO.md`, which orders them against each other and against the rest of the
work. What is left here is the five that still need a trained artifact, an
untested hypothesis, or a host this project has not seen.

The numbering is unchanged, because `CHANGES.md`, `TODO.md` and the source all
cite these ideas by number. A moved or built idea keeps its slot as a stub with
its citation.

## Bottom line

- **The kernels are finished and the memory is the wall.** 0.8.9 put every
  width the export packs at the memory's own rate. On the reference host that
  wall is **41 tokens a second** — 784.4 MiB a step against a 32.18 GiB/s
  sweep — and decode is at 12.57 of it. Closing that gap is measurement and
  scheduling work, and it is in `TODO.md`. **Passing** it needs more than one
  token per sweep, which is speculation, and that is in `TODO.md` too.
- **30 ms for a fresh, full-resolution image is still a different scale of
  problem.** A picture costs 48.3 s at one thread today, about 32 of it in the
  tower. Tiling the tower's attention and cutting patches before the pooling —
  both in `TODO.md` — are worth large factors and not three orders of
  magnitude. A distilled encoder is the only route on this list that could be,
  and it needs training.
- **The remaining research bets are QAT-cell-certified skipping (7) and a
  drafter conditioned on this model's per-layer inputs (8).** Both are
  hypotheses. A limited search cannot establish that nobody has tried them, and
  neither is claimed as novel.

Labels used below:

- **Established:** supported by prior work; porting and measuring remain work.
- **Adaptation:** a proposed engine-specific combination of known techniques.
- **Hypothesis:** an unvalidated research direction.
- **Model-preserving:** targets the same model computation, not necessarily
  bit-identical floating-point results. **Approximate** changes behavior.
  **Training required** needs a new artifact, not merely engine code.

## 1. What this engine gives a researcher

| Observation | Why it matters |
| --- | --- |
| The text configuration has 35 layers, width 1536, vocabulary 262144, and `enable_moe_block: false`. | Generic MoE routing ideas do not accelerate this checkpoint just because the engine supports MoE. |
| The output head is **untied** and 2-bit: 262144 by 1536 is 96.0 MiB, 12.2% of what a decode step reads. | Bounding the head is worth a real fraction here, unlike a checkpoint that ties it to the embedding table. |
| Every code plane ships an `input_activation_scale`, and activations reaching one are exactly `level * step` for an int8 level. | This is what 0.8.9 built on, and it is also the premise any certified-skip bound would need. |
| The checkpoint has **no** `activation_sparsity_pattern`. Gemma 3n has one and llama.cpp implements it; this export does not. | Idea 9's exact-zero route has no architectural support here and must earn its keep on measured occupancy alone. |
| `session_pass` returns logits for only its last lane. There are sliding rings and cross-layer KV sharing. | Speculative verification is not just feeding a longer prompt: it needs all-position outputs and transactional state. |
| The vision tower has 16 layers, width 768, 12 heads, 16-pixel patches, and 3×3 pooling **after** the encoder. | Reducing tokens before or inside the tower removes expensive work; reducing only the final soft tokens cannot. |
| Grid selection targets the configured patch budget, including upscaling small inputs. | Supplying a smaller JPEG does not by itself request less encoder computation. |

Evidence: `model/config.json`, `model/processor_config.json`, and in
`app_core.c` the plane binding, `plane_lift_many`, `kern_level_stage`, the
session pass and the tower.

**Already built, and not to be sold again as research:** SIMD and unpack tuning
at every width; shared-row GQA cache reads; the byte KV cache; batched prefill;
the softmax series; prompt and session persistence; the pool's spin; and — as
of 0.8.9 — integer matrix kernels on the calibrated grid, for both the single
lane and the batched path.

## 2. Feasibility before enthusiasm

### Text: the ceiling is measured, not estimated

A decode step reads **784.4 MiB** as the engine tallies it (759.4 MiB of weights
and tables, the rest cache and working data). For an ordinary dense decode with
no cross-token weight reuse:

> tokens/s ≤ sustained memory bandwidth ÷ bytes read per token

On the reference host — four cores of a Xeon at 2.8 GHz, AVX-512 and VNNI — a
bare sweep gives 32.18 GiB/s at four threads, so the ceiling is **41 tokens a
second**. The engine is at 12.57. Other hosts differ and their numbers must not
be combined; a many-channel server CPU has a different ceiling, and "a CPU" is
not a hardware specification.

Reaching 100 tokens a second on this host is therefore not a kernel question at
all. It needs either far more memory bandwidth, or more than one committed token
per sweep. Speculation amortizes the target's weights across sequential output
tokens; batching independent users amortizes them across users, which does not
serve the one-conversation goal. Speculation still pays the arithmetic for
verification: if a round costs 35 ms and commits 4 tokens it yields about 114
tokens/s; at 45 ms and 2 committed tokens it yields 44. Those are
**illustrations**, not forecasts, and a round's time must include drafting,
every verified position's output head, correction and rollback.

### Vision: define the boundary before quoting a number

Measure separately: (1) decode, resize, normalization and patch preparation;
(2) encoder plus projector on a fresh image; (3) text-stack prefill of the
visual rows; (4) time to first answer token. "30 ms vision" should mean 1 + 2
unless stated otherwise, and never implies a 30 ms multimodal answer.

The current measured baseline, one thread, a 768 by 768 image at the full patch
budget: **48.3 s** for the whole picture, of which **16.3 s** is (3) and about
**32 s** is (2). Four threads takes the whole to 20.7 s. Reaching 30 ms for (2)
from 32 s is a factor of a thousand. Nothing that keeps this encoder at this
resolution is going to find it.

The advertised budget of 280 soft tokens is not 280 encoder patches: a
full-budget image is 48×48 = 2304 patches, pooled 3×3 to 256 rows **after** the
encoder has paid for all of them. Any claim using the output row count as the
encoder sequence length is misleading.

## 3. What moved, and where it went

These six are in `TODO.md` now, ordered against the rest of the work, each with
what reading `llama.cpp` said about it. Their slots here keep the citations.

| # | Idea | Where it is now |
| --- | --- | --- |
| 1 | CPU-adaptive speculative verification [1] | `TODO.md` — the only route past the memory ceiling. llama.cpp has three forms of it. |
| 4 | Explicit adaptive image budgets | `TODO.md`, with idea 5. llama.cpp exposes the budget and has no escalation policy. |
| 5 | Pool-aware early token compression [3], cf. FastV [4] | `TODO.md` — the largest vision lever; nothing like it in llama.cpp's `clip.cpp`. |
| 10 | Certified vocabulary screening | `TODO.md` — 12.2% of a step, and not in mainline llama.cpp. |
| 11 | Tiled online vision attention [8] | `TODO.md` — llama.cpp has this on CPU, which is the argument for it. |
| 12 | Visual feature reuse by content identity | `TODO.md` — exact and narrow. llama.cpp reuses a picture only through a prompt-prefix KV match inside one slot; there is no content-addressed store of projected rows. |

### 2. Compute on the calibrated integer grid — **built, 0.8.9**

Done. Activations reaching a code plane are `level * step` for an int8 level, so
`Σ code·act` is `step` times an exact integer; one `vpdpbusd` takes 64 of those
products where the float loop unpacked, converted and multiplied 16. All three
widths, both the single-lane and the batched path. The staging certifies itself
against the grid and falls back where a caller did not round. `CHANGES.md` 0.8.9
has the kernel rates, the engine numbers and the reference comparison.

What is left of it is width portability — AVX-VNNI at 256 bits, ARM `sdot`, and
the AMX tile path llama.cpp also carries — which is an entry in `TODO.md`.

## 4. What is still research

### 3. T-MAC-style low-bit lookup execution

**Established. Model-preserving in intent or approximate depending on the LUT
arithmetic. No training.**

T-MAC [2] avoids low-bit dequantize-and-multiply by building activation lookup
tables and using packed weight bits to index them.

**What 0.8.9 changed about this idea.** It answers the same question the integer
path answered — how to stop spreading codes into floats — so on a host with an
integer dot product there is nothing left for it to win: that host is already at
the memory. Its remaining case is a host with neither AVX-VNNI nor ARM `sdot`,
where it is the only route to the same place. That narrows it from a headline to
a portability experiment, and `TODO.md` carries it there.

Not in mainline llama.cpp; T-MAC is a separate fork.

**Failure modes:** table construction dominates small matrices; lookup precision
changes results; larger tables evict useful data; a bitplane layout needs
packing. It saves instructions, not the weight-byte bandwidth floor — so on any
host where the integer path already runs, its ceiling is zero.

**Experiment:** total table-build-plus-matmul time and full decode speed on a
host with no integer dot product. Reject isolated kernel wins that lose in the
engine.

### 6. Distil a smaller encoder into this model's visual interface

**Established direction. Approximate. Training required.**

Use a compact CPU-friendly hierarchical encoder, in the spirit of FastViT or
MobileCLIP [5,6,9], and train a projector to match the teacher's visual
representations and downstream answers. A classification or CLIP embedding is
not a drop-in replacement for this model's spatial soft-token sequence.

**Why this is here and not in `TODO.md`.** It is the only idea on either list
that could plausibly reach a 30 ms fresh image, because it is the only one that
removes orders of magnitude rather than factors. It is also the only vision idea
that gives up unchanged-checkpoint parity outright and needs training resources
this project does not have.

Train against pooled and projected teacher features as well as task losses.
Retain the row count and ordering initially, to isolate encoder replacement from
row-count reduction. Include documents, fine text, diagrams and small objects in
supervision.

**Experiment:** a latency and quality Pareto curve for several students on the
actual target CPU, including preprocessing and projection. Mobile-device or
image-classification results in the cited papers do not prove multimodal CPU
latency here.

### 7. QAT-cell-certified partial evaluation

**Hypothesis. Model-preserving only with sound bounds. No training initially.**

The output grid makes some uncomputed arithmetic irrelevant. Evaluate a coarse
or partial dot product, bound the remaining contribution, and stop when the
**entire** possible result lies inside one output quantization cell: the final
quantized value is then fixed without evaluating the remainder.

**What 0.8.9 changed about this idea.** It made the premise concrete rather than
assumed. The engine now knows, per product, that its activations are exact int8
levels on a stated step, and it computes an exact integer sum — so a bound on
the remainder is a bound on integers, not on floats, and does not have to cover
the float reduction's own rounding error. That removes the hardest part of the
soundness argument. It also raises the bar: the thing to beat is now a kernel
running at the memory's rate, so a skip must avoid *reads*, not just multiplies.

Candidate bounds: blockwise residual norms or precomputed absolute weight sums
per row, combined with cheap activation summaries. Start with a projection that
immediately applies `leave_gain`, not a whole block. The bound must still cover
clipping, the group scales, ties-to-even and the final requantization; boundary
cases fall back to full evaluation.

**Main risk, restated for the post-0.8.9 engine:** high-dimensional bounds are
loose, and reading enough of a row to certify the result costs the row. Any
precomputed metadata is itself bytes the step must read, and the step is
bandwidth-bound, so metadata that is not much smaller than what it saves is a
regression rather than a wash.

**Experiment and stop rule:** on real activations, histogram how often a cell is
certified and measure **net bytes not read**. Stop if useful skips need most of
a row, or if the metadata cancels the saving. A successful version offers
something stronger than heuristic pruning: locally proven skips.

Not in mainline llama.cpp, which has no early-termination path of any kind.

### 8. A per-layer-input-conditioned, cache-sharing drafter

**Hypothesis. Exact target verification. Drafter training required.**

Train a tiny recurrent or shallow drafter on selected target hidden features and
this architecture's per-layer token inputs, with the objective being agreement
with the **quantized deployed target** rather than low perplexity against a
full-precision teacher.

**Why this is here and not in `TODO.md`.** The speculation entry in `TODO.md`
starts with an n-gram proposer, which needs no training and tests the verifier.
This idea is what to try if that proposer's acceptance is poor, and it needs an
artifact. Keep the verifier from that entry, so proposal quality affects speed
rather than correctness.

LayerSkip [7] motivates early-exit self-speculation, but skipping arbitrary
layers in this checkpoint is not automatically a trained early-exit model. The
per-layer embeddings and shared KV owners make the state interface more involved
than a generic shallow transformer. Use only features already available at the
drafting point; running missing target layers to obtain them defeats the saving.

**Experiment:** compare at equal draft-time budgets against n-gram lookup and an
ordinary small draft model. Track acceptance specifically on image-grounded
answers — a text-only drafter may miss visual facts.

### 9. Exploit zeros and changes structurally, not neuron by neuron

**Adaptation / hypothesis. Exact zeros or approximate deltas. Training
optional.**

Calibrated 8-bit activations can contain exact zeros, and skipping their
contributions is mathematically valid. Unstructured indexing may nevertheless
cost more than dense SIMD, and a transposed or tiled sparse-column strategy must
actually avoid touching the associated weights — which is the only thing that
matters on a bandwidth-bound step.

**What reading the checkpoint changed about this idea.** Gemma 3n ships an
`activation_sparsity_pattern` and applies architectural sparsity to its first
layers; llama.cpp implements it (`src/models/gemma3n.cpp`). **This export has
no such field.** So there is no architectural sparsity to exploit here, and the
idea has to earn its keep on measured occupancy of incidental zeros — a much
weaker starting position than the Gemma 3n case suggests. Measure the histogram
before building anything.

PowerInfer [10] is relevant prior work on activation locality, but its
CPU/GPU heterogeneous execution is not CPU-only evidence for this GELU
checkpoint. GELU is not ReLU and near-zero is not zero.

The delta variant — cache a previous projection and compute an activation delta
for a similar input, as in successive video frames — changes the summation and
must use the post-quantization input. Similar tokens do not imply similar hidden
states, and the tower's attention is global, so a small pixel change reaches
every patch.

**Experiment and stop rule:** histogram exact-zero block occupancy on real
activations first. Stop if dense kernels win after indexing, packing and the
extra cached-state traffic — which, at the memory's rate, they very well may.

## 5. Primary research anchors

These sources support mechanisms, not performance guarantees for igllm.
Publication dates and experimental hardware matter; the newest paper is not
necessarily the best fit. This is a bounded literature pass, not an exhaustive
novelty review.

1. **Fast Inference from Transformers via Speculative Decoding** — Leviathan,
   Kalman, Matias; 2022 preprint / ICML 2023.
   https://arxiv.org/abs/2211.17192
2. **T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment on
   Edge** — 2024 preprint. https://arxiv.org/abs/2407.00088
3. **Token Merging: Your ViT But Faster** — Bolya et al.; 2022 preprint /
   ICLR 2023. Encoder token merging, not lossless pooling reordering.
   https://arxiv.org/abs/2210.09461
4. **An Image is Worth 1/2 Tokens After Layer 2** — 2024; FastV. Operates on
   visual tokens inside the language model; it does **not** remove the preceding
   encoder cost. https://arxiv.org/abs/2403.06764
5. **FastViT: A Fast Hybrid Vision Transformer using Structural
   Reparameterization** — 2023. https://arxiv.org/abs/2303.14189
6. **MobileCLIP: Fast Image-Text Models through Multi-Modal Reinforced
   Training** — 2023 preprint / CVPR 2024. https://arxiv.org/abs/2311.17049
7. **LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding**
   — 2024. Training and inference co-design, not guaranteed plug-in early exits.
   https://arxiv.org/abs/2404.16710
8. **FlashAttention: Fast and Memory-Efficient Exact Attention with
   IO-Awareness** — 2022. Original results are GPU results; llama.cpp's CPU
   adaptation is the closer precedent. https://arxiv.org/abs/2205.14135
9. **MobileCLIP2: Improving Multi-Modal Reinforced Training** — 2025.
   https://arxiv.org/abs/2508.20691 · https://github.com/apple/ml-mobileclip
10. **PowerInfer: Fast Large Language Model Serving with a Consumer-grade
    GPU** — 2023 preprint. Heterogeneous sparse execution, not a CPU-only
    result. https://arxiv.org/abs/2312.12456

**Novelty boundary:** ideas 5, 7, 8 and the certified/delta combination in 9 are
proposed engine-specific experiments. The sources do not establish those
combinations as successful here, and this document does not claim they are
unpublished or patentably novel. The llama.cpp comparisons throughout are a
reading of one tree at one commit, not an audit of the field.
