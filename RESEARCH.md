# Beyond the TODO: CPU text and vision acceleration

Research snapshot: 2026-09-06. This is an experiment shortlist, not a performance
claim. Assume every current TODO is complete; none of the historical timings
below is a prediction of that finished engine. No new benchmarks were run.

## Bottom line

- **100–200+ tokens/s in one conversation needs more than faster unpacking.**
  On the documented desktop memory bandwidth, the strongest unchanged-model
  route is accepting several speculative tokens per target weight sweep.
  Better kernels are the foundation, not a way around the bandwidth limit.
- **30 ms for a fresh, full-resolution image is a different scale of problem.**
  The credible CPU route combines much less encoder work with integer matrix
  kernels, probably a distilled encoder and adaptive resolution. Maintaining
  the existing full-budget model's behavior on an ordinary desktop is not a
  defensible promise.
- My highest-value *research* bets are **QAT-cell-certified computation
  skipping**, **pool-aware early vision compression**, and **a small drafter
  conditioned on this model's per-layer inputs**. These are hypotheses,
  not claims of first-ever invention or proven speedups.

Labels used below:

- **Established:** supported by prior work; porting and measuring remain work.
- **Adaptation:** a proposed engine-specific combination of known techniques.
- **Hypothesis:** an unvalidated research direction. A limited search cannot
  establish that nobody has tried it.
- **Model-preserving:** targets the same model computation/distribution, not
  necessarily bit-identical floating-point results. **Approximate** changes
  behavior. **Training required** needs a new artifact, not merely engine code.

## 1. What this engine actually gives us

These observations concern the inspected checkout, not the other agent's future
changes. Absolute source paths are included to make the reasoning auditable.

| Observation | Why it matters |
| --- | --- |
| The shipped text configuration has 35 layers, width 1536, vocabulary 262144, and `enable_moe_block: false`. | Generic MoE routing ideas do not accelerate this checkpoint just because the engine supports MoE. |
| Text weights mix 2/4/8 bits; vision weights are 8-bit. Calibrated projections round activations to an 8-bit grid. | Integer activation computation can exploit quantization already present, rather than imposing another arbitrary precision loss. |
| `kern_row_code_many` spreads weights into floats, then dots them against lanes. `plane_lift_many` stages calibrated activations in chunks of at most 16. | There is a concrete starting point for a different LUT/integer execution strategy; batching itself already exists. |
| `session_pass` returns logits for only its last lane. There are sliding rings and cross-layer KV sharing. | Speculative verification is not just feeding a longer prompt: it needs all-position outputs and transactional state. |
| The vision tower has 16 layers, width 768, 12 heads, 16-pixel patches, and 3×3 pooling **after** the encoder. | Reducing tokens before/inside the tower can remove expensive work; reducing only the final soft tokens cannot. |
| Grid selection targets the configured patch budget, including upscaling small input images. | Supplying a smaller JPEG does not by itself request less encoder computation. |

Evidence:

- `/home/runner/work/igllm/igllm/model/config.json:55–103,104–186,191–234`
- `/home/runner/work/igllm/igllm/model/processor_config.json:25–47`
- `/home/runner/work/igllm/igllm/app_core.c:1552–1570,2560–2595,5500–5534`
- `/home/runner/work/igllm/igllm/app_core.c:7341–7369,7400–7424,7511–7515,9185–9270`

Already assumed solved or already implemented: ordinary SIMD/unpack tuning,
shared-row GQA cache reads, byte KV cache, batched prefill, faster softmax,
prompt/session persistence, tensor residency, and the backend handle TODO.
Do not sell these again as new research.

## 2. Feasibility before enthusiasm

### Text: distinguish single-stream latency from aggregate throughput

The repository reports about **759.4 MiB of weight/table traffic per decode
step**, or **0.796 GB**, excluding growing KV traffic and other working data.
The more narrowly quoted **727.5 MiB** is packed code traffic, not the whole
step. Neither is the total mapped model size.

For an ordinary dense decode without effective cross-token weight reuse:

**tokens/s ≤ sustained memory bandwidth / bytes read per token**

| Target | Weight/table bandwidth alone |
| --- | --- |
| 100 tokens/s | 79.6 GB/s |
| 200 tokens/s | 159.3 GB/s |
| 300 tokens/s | 238.9 GB/s |

At the README's historical 24.9 GB/s sequential-read measurement, the idealized
ceiling is about **31 tokens/s** before computation, attention, and scheduling.
Other hosts in TODO use GiB/s and have different results: do not combine their
numbers into a claimed measured baseline. A many-channel server CPU can have a
different ceiling; “a CPU” is not a hardware specification.

Speculation amortizes target weights across sequential output tokens; batching
independent users amortizes them across users. Only the former directly serves
the one-conversation goal. Speculation still pays the arithmetic for verification.

If a speculative round costs 35 ms and commits 4 tokens on average, it yields
about 114 tokens/s. At 45 ms and 2 committed tokens, it yields only 44 tokens/s.
These are **illustrations**, not forecasts. Include drafting, every verified
position's output head, correction, and rollback in the round time.

### Vision: define the 30 ms boundary

Measure separately:

1. Image decode, resize, normalization, and patch preparation.
2. Vision encoder plus projector, on a fresh image.
3. Text-model prefill of the visual rows and question.
4. Time to first answer token and subsequent decode.

“30 ms vision” should mean **1 + 2** unless explicitly stated otherwise.
It does not imply a 30 ms multimodal answer.

TODO's historical full-budget example is a **48×48 = 2304-patch** image and
describes roughly **600 billion projection multiply-adds**. Taking that rough
count literally would require **20 trillion MAC/s sustained** for projections
alone at 30 ms, or 40 TOPS if one MAC counts as two operations. Recompute the
exact count from bound tensor shapes before using it as a benchmark denominator.
The recorded 113.2-second image result would need roughly a **3770×** reduction
to reach 30 ms; it is a historical single-thread run, not the post-TODO baseline.

The advertised budget of 280 soft tokens is not 280 encoder patches. Pooling
reduces the example's 2304 patches to 256 output rows only after the heavy work.
Any claim using the output row count as the encoder sequence length is misleading.

Sources for these historical measurements:
`/home/runner/work/igllm/igllm/README.md:230–265` and
`/home/runner/work/igllm/igllm/TODO.md:11–54`.

## 3. Ranked shortlist

Priority is experiment order, not an estimated implementation duration. Even
established techniques below need measurement against the **finished TODO**
baseline; discard anything that merely reproduces its gains.

| Priority | Idea | Status | Behavior / training | Main opportunity |
| --- | --- | --- | --- | --- |
| 1 | CPU-adaptive speculative verification | Established + adaptation | Preserving with correct verifier; training optional | Single-stream text throughput |
| 2 | QAT-native integer matrix kernels | Established + adaptation | Preserving intent; numerical validation needed; no training | Vision and verification compute |
| 3 | Low-bit lookup-table execution | Established | Preserving intent or approximate, depending on LUT arithmetic; no training | Text decode instructions |
| 4 | Explicit adaptive image budgets | Established + adaptation | Approximate; initially no training | Fewer encoder patches |
| 5 | Pool-aware early token compression | Adaptation / hypothesis | Approximate; fine-tuning likely | Large vision work reduction |
| 6 | Distilled compact vision replacement | Established direction | Approximate; training required | Strongest fresh-image 30 ms route |
| 7 | QAT-cell-certified partial evaluation | Hypothesis | Preserving only with sound bounds; no training initially | Skip provably irrelevant work |
| 8 | Per-layer-input-conditioned drafter | Hypothesis | Exact target verification; drafter training required | Cheap, high-acceptance proposals |
| 9 | Structured zero/delta execution | Adaptation / hypothesis | Exact zeros or approximate deltas; training optional | Avoid weight reads and MACs |
| 10 | Certified vocabulary screening | Adaptation | Preserving greedy/top-k only under stated proof; no training | Output-head cost |
| 11 | Tiled online vision attention | Established | Preserving intent; no training | Attention traffic and workspace |
| 12 | Visual feature reuse / temporal refinement | Established + adaptation | Exact whole-image reuse; approximate partial reuse | Repeat images and video |

### 1. CPU-adaptive speculative verification

Use a cheap proposer to predict a short block, then verify the block with the
target model's batched path. Start with prompt lookup or a small n-gram proposer
to test the verifier without training; use a compact draft model if acceptance
on ordinary questions is poor. Speculative decoding [1] is the established
principle; the CPU-specific opportunity is choosing block length by actual
**committed tokens per millisecond**, not acceptance rate alone.

**Fit:** extend `session_pass` to provide the target distribution at every
verified position, with correctly shifted token alignment. Include the large
output head in batched verification rather than repeatedly streaming it.

**Correctness trap:** rejected proposals can overwrite old sliding-window KV
rows. Restoring only `fill_count` is insufficient. Use transactional scratch
or undo data for overwritten owner rows, shared-cache relationships, token
history, and sampling state. Do not serialize an entire session per round.
Greedy verification is simpler; stochastic exactness needs the acceptance and
residual-distribution procedure, with temperature, penalties, and filtering
applied consistently. Equality of distributions does not promise the same
seeded sequence when RNG consumption changes.

**Experiment:** blocks of 2, 4, and 8 on prose, code, multilingual text, and
image-conditioned answers; report accepted length, draft/verify/rollback time,
and p50/p95 output latency. Disable speculation when it loses. Large gains
require high acceptance **and** fast multi-lane matrix work.

### 2. Compute on the calibrated integer grid

The engine currently represents already-quantized activation levels as floats.
Retain integer levels through eligible projections, accumulate integer dot
products, then apply the actual input/weight scales and output-grid rounding.
For the int8 vision tower, target CPU dot-product/tile facilities where present:
AVX-VNNI/AVX-512 VNNI or AMX on appropriate x86 hosts, and dot-product/I8MM
facilities on appropriate ARM hosts. These are CPU capabilities, not GPU use.

Use a multi-row/multi-lane matrix microkernel rather than treating every output
row as an independent float dot. Keep shape-specific choices: decode, target
verification, and thousands of image patches have different best kernels.
If TODO already delivers equivalent integer kernels, this experiment is done.

**Constraints:** calibrated inputs are not proof that reassociated integer
arithmetic is bit-identical to the old floating reduction. Preserve signedness,
zero-point corrections, group scales, clipping, and ties-to-even; prove
accumulator bounds including bias corrections. Uncalibrated planes retain a
fallback. Validate QAT boundary cases, not just mean tensor error.

Optional backend-owned packing can improve tile access without modifying the
checkpoint format. Account for its extra resident memory, initialization time,
and departure from strict zero-copy operation. Keep portable C fallbacks and
check ISA plus OS support before dispatch.

**Experiment:** first one vision MLP and one target-verification batch, then
end-to-end. A peak integer TOPS figure is not measured tower throughput.

### 3. T-MAC-style low-bit lookup execution

T-MAC [2] avoids conventional low-bit dequantize-and-multiply by constructing
activation lookup tables and using packed weight bits to index them.
This fits the 2/4-bit text projections far better than another marginally wider
float unpack loop. It is not an automatic match for the 8-bit vision tower.

**Adaptation:** compare LUT decode against integer execution across bit width,
row width, and lane count; select the cheaper strategy. Reuse activation-side
tables across projections only when their input vectors **and calibrated grids**
are identical. Gate/up projections are candidates, not a guarantee.

**Failure modes:** LUT construction dominates small matrices; lookup precision
changes results; larger tables evict useful data; a bitplane layout needs
packing. This saves instructions, not the original weight-byte bandwidth floor.

**Experiment:** total table-build-plus-matmul time, cache misses, and full decode
speed on AVX2 and ARM separately. Reject isolated kernel wins that lose in engine.

### 4. Explicit image budgets, with escalation

Expose an actual patch/soft-token budget rather than merely shrinking the input
file. Start at a low grid for ordinary scene understanding; escalate resolution
for OCR, counting, small objects, or uncertain answers. Use a cheap selector or
the requested task, not a complete expensive pass just to choose a cheap pass.

At one quarter the patches, projection MACs are roughly one quarter and dense
attention pair work roughly one sixteenth. That is arithmetic scaling, not a
promised total speedup. Preserve valid 3×3 pool geometry, positions, and emitted
media-token counts.

**Experiment:** a quality/latency curve over supported grid budgets, split by
OCR versus coarse understanding. Report escalation frequency and p95 latency;
a 30 ms fast path with frequent second-scale fallbacks is not “30 ms images.”

### 5. Move compression earlier, but respect the pool geometry

ToMe [3] establishes that merging similar ViT tokens can reduce encoder work.
The engine-specific hypothesis is **pool-aware adaptive merging**: spend the
first few blocks on the full grid, then collapse redundant members of eventual
3×3 pooling cells while retaining detail tokens where needed.

Represent token mass and original spatial support; preserve enough provenance
to reconstruct the final spatial pooling contract. Test merge depths and
retained-token counts, not just one fixed compression ratio.

**Why this is not a safe algebraic rewrite:** averaging does not commute with
attention, GELU, normalization, or 2-D rotary positions. A mass correction to
attention cannot recover discarded information. Treat it as approximate and
expect fine-tuning/distillation to be necessary for aggressive compression.
Protect text-like edges and small objects; saliency alone can miss counting.

**Experiment:** merge after layers 2/4/8 and measure downstream OCR, counting,
spatial relations, and VQA, including pooling reconstruction overhead.
FastV [4] is related but operates on visual tokens inside the language model;
it does **not** remove the preceding vision encoder cost.

### 6. Distill a smaller encoder into this model's visual interface

Use a compact CPU-friendly hierarchical encoder, inspired by FastViT or
MobileCLIP [5,6], and train a projector/adapter to match the teacher's visual
representations and downstream answers. A classification or CLIP embedding is
not a drop-in replacement for this model's spatial soft-token sequence.
MobileCLIP2 [9] is a verified 2025 continuation worth examining for its improved
reinforced-training recipe, not evidence that this encoder interface or CPU
latency target has already been solved.

Train against pooled/projected teacher features as well as task losses.
Retain the row count/ordering initially to isolate encoder replacement; reducing
the LLM-side row count is a separate adaptation needing its own quality checks.
Include documents, fine text, diagrams, and small objects in supervision.

**Why prioritize this:** a fresh-image 30 ms target likely requires removing
orders of magnitude of work, not optimizing the same full-budget transformer.
It gives up unchanged-checkpoint parity and requires training resources.

**Experiment:** latency and quality Pareto curve for several small students at
the actual target CPU, including preprocessing and projection. Mobile-device
or image-classification results in the cited papers do not prove multimodal
CPU latency here.

### 7. QAT-cell-certified partial evaluation

**Hypothesis:** the output grid makes some uncomputed arithmetic irrelevant.
Evaluate a coarse or partial dot product, bound the remaining contribution, and
stop if the **entire** possible result lies inside one output quantization cell.
The final quantized value is then fixed without evaluating the remainder.

Potential bounds include blockwise residual norms or precomputed absolute
weight sums, combined with cheap activation summaries. Start with a projection
that immediately applies `leave_gain`, not a whole transformer block.

To preserve behavior, the bound must cover the actual implementation's float
rounding error, input/weight scales, clipping, ties-to-even, and the final
requantization. A real-arithmetic bound alone does not certify the existing
floating implementation. Boundary cases fall back to full evaluation.

**Main risk:** high-dimensional bounds are too loose, and reading enough weights
to certify the result costs as much as computing it. Precomputed metadata must
be much cheaper than the avoided weight reads.

**Experiment / stop rule:** measure certified-cell frequency and **net bytes
avoided** on real activations. Stop if useful skips require nearly a full row
read or metadata overhead cancels the saved work. A successful version could
offer something stronger than ordinary heuristic pruning: locally proven skips.

### 8. A per-layer-input-conditioned, cache-sharing drafter

**Hypothesis:** train a tiny recurrent or shallow drafter using selected target
hidden features and this architecture's per-layer token inputs. The objective
is agreement with the **quantized deployed target**, rather than merely low
perplexity against a full-precision teacher.

LayerSkip [7] motivates early-exit/self-speculation, but skipping arbitrary
layers in this checkpoint is not automatically a trained early-exit model.
Per-layer embeddings and shared KV owners also make the state interface more
involved than a generic shallow transformer. Only use features already available
at the drafting point; running missing target layers to obtain them defeats
the saving.

**Experiment:** compare at equal draft-time budgets against n-gram lookup and
an ordinary small draft model. Track acceptance specifically on image-grounded
answers: text-only draft training may miss visual facts. Keep the full target
verifier from idea 1, so proposal quality affects speed rather than correctness.

### 9. Exploit zeros and changes structurally, not neuron by neuron

Calibrated 8-bit activations can contain **exact zeros**. Skipping their
contributions is mathematically valid; unstructured indexing may nevertheless
cost more than dense SIMD. A transposed/tiled sparse-column strategy must
actually avoid touching the associated weights.
PowerInfer [10] is relevant prior work on activation locality, but its
CPU/GPU heterogeneous execution and sparsity assumptions are not CPU-only
performance evidence for this GELU checkpoint.

**Adaptation:** group active columns into SIMD-friendly blocks and switch
between dense and sparse kernels based on measured occupancy. For repeated
video frames or similar states, a more speculative extension caches a previous
projection and computes an activation delta.

**Limits:** GELU is not ReLU, and near-zero is not zero. Delta updates can change
floating summation and must use the post-quantization input. Similar tokens do
not imply similar hidden states; global vision attention spreads local changes.
Approximate thresholding needs explicit quality budgets or certification from
idea 7. Training structured sparsity into a student is a separate option.

**Experiment / stop rule:** histogram exact-zero block occupancy and delta
density before building a large sparse runtime. Stop if dense kernels win after
indexing, packing, and extra cached-state traffic.

### 10. Certify that most vocabulary rows cannot win

The 262144×1536 2-bit output head alone contains about **96 MiB** of codes,
roughly 13% of the documented per-step traffic. Cluster output rows; compute
cheap upper bounds for each cluster, evaluate promising rows, and discard a
cluster only when its bound cannot beat the best candidate.

**Adaptation:** make bounds aware of the head's quantization and output
transforms. Monotonic softcapping preserves order, but quantization can create
ties; preserve the baseline tie-break, include numerical error margins, and
account for repetition penalties before declaring a winner.

**Scope:** exact greedy selection, or exact top-k if every excluded row is
provably below the retained threshold. Arbitrary softmax/top-p sampling cannot
simply ignore the discarded tail probability. ANN shortlists alone are
approximate.

**Experiment:** greedy benchmark first; measure bound tightness and head bytes
actually avoided. Even removing all head code traffic would improve the ideal
bandwidth ceiling only about 1.14×. This is a useful multiplier, not the main
100 tokens/s strategy.

### 11. CPU-tiled online attention for the vision encoder

FlashAttention [8] establishes IO-aware tiled attention with online softmax.
Adapt the principle to CPU cache sizes: compute score tiles, update rowwise
normalizers, and accumulate value outputs without repeatedly materializing
large attention intermediates.

This is not “use a faster exp” again. Its purpose is a different memory schedule,
with CPU-specific tile sizes and parallel ownership.

**Limits:** it does not remove the quadratic attention arithmetic or the much
larger projection cost reported historically. Floating reduction order changes.
The batched integer projections and token reduction above deserve higher
priority. Smaller working sets may become particularly valuable after those
improvements shift the bottleneck.

**Experiment:** compare full tower time, peak scratch memory, and numerical
drift at several grid sizes. Do not import a GPU FlashAttention speedup number.

### 12. Content-addressed visual features and temporal refinement

Cache **post-projector visual rows** by a collision-resistant content identity
and model/preprocessing configuration. This reuses image computation across
different questions or conversations, beyond an identical prompt-cache prefix.
Identity must include orientation, resize budget, quantization/backend behavior,
and any encoder adaptation. Bound memory and invalidate on model changes.
Store/cache media only under the application's privacy and retention policy.

Identical-image reuse can preserve results but is not fresh-image acceleration.
For video, reuse unchanged patch preparation and test an approximate refinement
path for changed regions. Full bidirectional attention means a small pixel
change can affect every patch, so reusing deep local activations is not exact
without stronger proof.

**Experiment:** report cold misses, warm hits, and temporal sequences separately.
Never report cache-hit latency as the 30 ms fresh-image result.

## 4. What I would actually try first

### Track A: keep the shipped target

1. Record a post-TODO roofline and a phase breakdown on the intended CPU.
2. Establish whether integer or LUT execution gives the better kernel for each
   shape, especially multi-lane verification and the int8 vision MLP.
3. Build the smallest correct speculative verifier and compare 2/4/8-token
   blocks. Add a trained drafter only after verification overhead is understood.
4. Spend the research budget on **QAT-cell certification** only if early trace
   analysis shows a realistic opportunity to avoid weight reads.

This is the most plausible unchanged-target route to 100+ text tokens/s.
It remains conditional on bandwidth, verification cost, and acceptance.

### Track B: prioritize 30 ms fresh-image latency

1. Establish a reduced-grid quality curve; merely resizing the source is not
   enough.
2. Compare early pool-aware compression with a compact distilled encoder.
3. Keep the selected encoder integer-native and its visual interface explicit.
4. Evaluate LLM visual-token prefill separately; if it dominates, investigate
   LLM-side pruning such as FastV with an explicit quality tradeoff.

For a strict 30 ms budget, a trained student is the first serious architectural
candidate. If unchanged full-resolution behavior is non-negotiable, change the
hardware/latency requirement rather than asserting an unsupported CPU speedup.

### Benchmark and acceptance contract

- **Hardware:** CPU model, physical cores/threads, ISA, memory channels/speed,
  thread affinity, power mode, sustained bandwidth, and thermal steady state.
- **Text:** single conversation, fixed prompt/context lengths, 256+ generated
  tokens where applicable, greedy and sampled runs, TTFT, committed tokens/s,
  inter-token p50/p95, and draft acceptance. Report aggregate serving separately.
- **Vision:** fresh varied images, grid dimensions and input format, cold versus
  warm runs, preprocessing/tower/projector/text-prefill timings, p50/p95, and RSS.
- **Quality:** existing synthetic/reference parity for preserving-intent changes;
  additionally assess logit drift, greedy disagreements, and QAT ties. Evaluate
  approximate paths on held-out OCR, counting, grounding, diagrams, multilingual
  prompts, and ordinary VQA. Choose permitted quality loss before tuning.
- **Speculation correctness:** rejection at every block position, ring wrap,
  shared KV owners, EOS, context limits, penalties, and distributions under
  sampling. A successful greedy demo is not evidence of exact sampling.
- **Attribution:** one change at a time and then combined; include initialization,
  extra metadata/packing, fallback frequency, and regressions. Do not multiply
  isolated speedups or hide poor p95 behavior behind an easy-image mean.

Do not implement all twelve ideas at once. The cheap deciding experiments are
verification cost, integer/LUT crossover, and the image budget/quality curve.
They determine whether the ambitious research directions have room to matter.

## 5. Primary research anchors

These sources support mechanisms, not performance guarantees for igllm. Publication
dates and experimental hardware matter; the newest paper is not necessarily the
best fit. This is a bounded literature pass, not an exhaustive novelty review.
Primary author repositories and publisher metadata were used where direct paper
retrieval was unavailable. MobileCLIP2 provides a verified 2025 lead; no 2026
work was sufficiently verified in this pass to recommend. That is a search
limitation, not a claim that no newer work exists.

1. **Fast Inference from Transformers via Speculative Decoding** — Leviathan,
   Kalman, Matias; 2022 preprint / ICML 2023.
   https://arxiv.org/abs/2211.17192
2. **T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge** — 2024
   preprint. Lookup-table mixed-precision matrix multiplication on CPUs.
   https://arxiv.org/abs/2407.00088
3. **Token Merging: Your ViT But Faster** — Bolya et al.; 2022 preprint /
   ICLR 2023. Encoder token merging, not lossless pooling reordering.
   https://arxiv.org/abs/2210.09461
4. **An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play Inference
   Acceleration for Large Vision-Language Models** — 2024; FastV.
   https://arxiv.org/abs/2403.06764
5. **FastViT: A Fast Hybrid Vision Transformer using Structural
   Reparameterization** — 2023.
   https://arxiv.org/abs/2303.14189
6. **MobileCLIP: Fast Image-Text Models through Multi-Modal Reinforced
   Training** — 2023 preprint / CVPR 2024.
   https://arxiv.org/abs/2311.17049
7. **LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding**
   — 2024. Training and inference co-design, not guaranteed plug-in early exits.
   https://arxiv.org/abs/2404.16710
8. **FlashAttention: Fast and Memory-Efficient Exact Attention with
   IO-Awareness** — 2022. Original results are GPU results.
   https://arxiv.org/abs/2205.14135
9. **MobileCLIP2: Improving Multi-Modal Reinforced Training** — 2025;
   models released August 28, 2025. A recent compact-encoder training direction.
   https://arxiv.org/abs/2508.20691
   Author implementation: https://github.com/apple/ml-mobileclip
10. **PowerInfer: Fast Large Language Model Serving with a Consumer-grade
    GPU** — 2023 preprint. Heterogeneous sparse execution, not a CPU-only result.
    https://arxiv.org/abs/2312.12456

**Novelty boundary:** ideas 5, 7, 8, and the certified/delta combination in 9
are proposed engine-specific experiments. The sources do not establish those
combinations as successful here, and this document does not claim they are
unpublished or patentably novel.
