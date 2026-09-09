# TODO

Open items, most consequential first. Each is what is left to do and why it is
worth doing — nothing here is progress, rationale or archaeology.

Everything closed is in `CHANGES.md` under the version that closed it.
Everything measured and refused is in that file's **refusal register**, along
with the hosts every number is quoted against and the shape of what a token, a
picture and a clip cost. Read those before reopening anything here.

Items marked **blocked** are scoped and cannot move on the hardware this project
has. Items marked **decision** are measured wins that are not taken, because
taking them would change what the engine outputs; they need a person to say so
in `CHANGES.md`, not a kernel.

---

## Engine

1. **Find where four fifths of the batched lane goes.** The largest single
   number in this file. A lane of the feed-forward is 991 M multiply-adds, which
   `vpdpbusd` should retire in about 2 ms over four threads; the measured lane is
   **8.9 ms** against a 45.7 ms step, and half of a speculative round is this one
   number. Raising it takes the speculative ceiling from 2.2x toward 4x and would
   move free generation from a cost to a wash. The four hypotheses that can be
   reasoned about from the source have all been measured and all came back small
   — the epilogue was the last, taken in 0.9.9 for a tenth of the lane.
   **Blocked:** take a hardware counter to it before writing another loop, and
   the reference host exposes no PMU.

2. **Merge redundant patches inside the vision encoder.** The pooling happens
   *after* the encoder, so every patch is paid in full and the pooling saves only
   the text stack's share. Merge after the first few blocks, keeping enough
   provenance to reconstruct the pooling contract. Its case is the caller who
   will *not* accept fewer patches up front: full resolution where the picture
   needs it and merging where it does not, inside one pass. Strictly harder than
   the patch budget and worth less than it was, since a caller who will accept
   fewer can now ask for fewer. Unexplored on CPU — no merging or pruning pass in
   llama.cpp's `clip.cpp`.

3. **Give the scout a real `q`.** A point mass is the strongest possible proposal
   and therefore the harshest: it stakes everything on one token, so acceptance
   under a temperature can never exceed `p(t)`. A proposer offering a
   distribution — a small draft model, or the scout's own counts turned into one
   — is accepted with `min(1, p/q)`, which can be 1 over a whole region rather
   than only at a near-certain token. The rejection rule this needs is already
   built and already tested; only the proposer is missing.

4. **Block the key axis in the tower's attention.** Four queries share the key
   row and nothing shares the query. A block of keys against a block of queries
   is the register-blocked matrix product this loop really is, and it is the only
   route left here that does not change the summation order. Scoring and the
   blend run at roughly twice their pre-0.9.6 rates and are still a long way
   under a 256-bit FMA peak of 44.8 G a core. The softmax is 6.6% of the phase
   and is not worth opening.

5. **Let a wound-back session survive a ring that has turned over.** After 0.9.14
   this is the binding limit on multi-picture caching, not the stamp.
   `session_hold` refuses once a layer's ring has wrapped, because slot `i` is id
   `i` only while it never did — so the rows cannot be wound back with the count.
   On the shipped export that is 512 ids, which two pictures at the checkpoint's
   own 280 rows exceed, so such a prompt keeps nothing whatever it shows. What
   would lift it is a ring that records where it wrapped, or a wind back that
   rewrites the slots rather than refusing. Neither is scoped. The cheap answer
   meanwhile is `--image-tokens` keeping the prompt under the window.

6. **Reach a turn whose prompt does not begin the same way.** A picture in the
   middle of a sentence — which `--text` allows and the content list means — puts
   the words in front of the rows, so the shared prefix ends before the expensive
   part. The media store still saves the tower; the prefill is paid again.

7. **Carry the integer dot product to its other widths.** 0.8.9's path is
   AVX-512 VNNI and nothing else. A host with `avx_vnni` and no AVX-512 —
   everything from Alder Lake on — has the same instruction at 256 bits; an ARM
   host has `sdot`/`udot` at 128; llama.cpp also carries an AMX tile path, which
   is a fourth width nothing here has looked at. **Blocked** on a host with the
   instruction and the headroom to show it, which is what 0.8.7 waited for on
   AVX-512.

8. **Try the order-preserving row block on AVX-512.** Thirty-two registers hold
   sixteen accumulators, four row vectors and two lane vectors with room over, so
   the batched float block that does not fit on AVX2 is available there. **This
   is a hypothesis reasoned on a host with no AVX-512** and is worth an hour on
   one that has it — with the caveat that where VNNI carries most planes onto the
   integer path, the float many path is a smaller share of a step.

9. **Read fewer bytes for a token in the feed-forward.** 52% of a step at 78% of
   the reference host's bare sweep, with both obvious kernel routes closed. This
   export is dense — `num_experts` is null in its `config.json` — and 35 layers
   of 1536 by 6144 gate, up and down is simply the tensor, so this is a sparsity
   or a residency question and not a loop one. **Do not reopen it as a kernel on
   a host this slow:** the third host sweeps at 49.80 GiB/s, where the same plane
   sits at 45% of a sweep rather than 78%, and whatever is left in the loop would
   show there and cannot show here.

10. **Let the caller choose residency per tensor.** Two thirds of the 2334.8 MiB
    the export maps is never touched by a token — the per-layer embedding table
    at 1120 MiB and the two towers at 324 — against the 759.4 MiB a step reads.
    A footprint question, worth having on a small host, and not a decode win
    however it is scoped. The export's other bf16 belongs here too: quantizing it
    at load is a footprint change as much as a speed one, and `ple lift` at
    12.5 G multiply-adds a second cannot join the integer path without a
    calibrated step it does not have.

11. **Measure a photograph of a page, and decide whether a budget can be chosen
    automatically.** The patch budget's quality curves are a shape rather than a
    study: two synthetic rasters, one prompt each, greedy. A photograph of a page
    is the case that decides what a caller should ask for. Any automatic policy
    has to hold off the floor below which the model reports no picture at all
    rather than a coarse one — *"Please provide the picture you are referring
    to."* — and cannot detect having crossed it from the answer. llama.cpp
    exposes the same knob with no escalation policy either; a fast path that
    falls back to full resolution half the time is not a fast path.

12. **Carry a block over a prompt that ends in a soft token.** Such a prompt
    takes the plain loop whatever `--guess` says, because a block cannot carry an
    embedding row. Small and named, and the last thing left of the speculative
    plumbing.

13. **Fuse `ple feed`'s gate, gelu and lift into one fork.** Saves 35 of the 70
    forks a step, but they are sequential — the lift needs the whole gate — so it
    needs a barrier inside a job, which `pool_group` does not have. Those 35
    forks are now about 0.02 ms a step. **Do not spend the barrier on them;**
    spend it only if something else wants one too.

14. **Run two conversations at once rather than one after the other.** The
    sessions are already independent; every kernel under them reaches the model's
    one `pool_group`, which is a fork and join with no queue and one caller's to
    be inside at a time — and 0.8.10 put the attention's two jobs and the gelu's
    behind that gate as well. It wants either a pool a session owns, or a queue
    in front of the one pool, which is the shape a server wants anyway. Not worth
    guessing at without a caller that needs it.

15. **Add a device handle beside `plane` and a second `back_open`,** so an
    accelerator backend drops in without touching the loader. It is also where a
    backend that repacks a plane at open would keep that packing, which every
    kernel experiment above would want. Written as an accelerator item; it is a
    portability one first.

16. **Take the tower blend's fused multiply-add.** *Decision.* Worth another
    0.6x — 2.2x fused against 1.4x not, on the AVX-512 host — and refused because
    it drops the intermediate rounding, so a picture's rows stop being the rows
    that ship. Taking it means saying so in `CHANGES.md` and re-taking the media
    parity run. The test that holds the kernel bit for bit fails on it
    deliberately. It is also the *worse* kernel on plain AVX2, where sixteen
    accumulators and four value registers do not fit in sixteen registers.

17. **Take the 1.33x batched float row block.** *Decision*, and the same shape as
    the one above: the only row block that pays changes the summation order, so
    prefill logits stop being byte for byte what they were. Not taken, and it
    should not be taken quietly if it ever is.

18. **Build under MSVC.** The suite builds clean and passes on Windows with MinGW
    gcc on the scalar, SSE2 and AVX2 backends; `cl` and its `/arch:AVX2` and
    `/arch:AVX512` paths have still only been read. **Blocked** without a Build
    Tools install.

19. **Hold the seam open as the prompt grows.** All nine cases are judged on the
    shipped export, each in its own process, and the longest is 664 ids with two
    pictures and a clip. What has not been tried is a prompt long enough that the
    reference's forward stops fitting even alone — the cost is the reference's,
    and the answer is probably to compare against a cached forward rather than a
    live one.

20. **The jpeg frames still refused.** Lossless, differential, hierarchical and
    arithmetic-coded, each another entropy coder or another frame shape rather
    than another branch. None of them is what a caller with a photograph has, so
    this is completeness rather than use.

---

## Research

What is above is buildable now against the shipped export. What is below needs a
trained artifact, an untested hypothesis, or a host this project has not seen.
The numbering is `RESEARCH.md`'s, kept so that citations in `CHANGES.md` and in
the source still resolve; that file has been folded into this one.

Labels: **established** — supported by prior work, porting and measuring remain;
**adaptation** — a proposed engine-specific combination of known techniques;
**hypothesis** — unvalidated. **Model-preserving** targets the same computation,
not necessarily bit-identical results; **approximate** changes behaviour;
**training required** needs a new artifact rather than engine code.

R1. **Idea 6 — distil a smaller encoder into this model's visual interface.**
*Established direction. Approximate. Training required.* The only idea on either
list that could plausibly reach a 30 ms fresh image, because it is the only one
that removes orders of magnitude rather than factors: a picture's encoder is tens
of seconds and the target is three orders below that. Use a compact CPU-friendly
hierarchical encoder in the spirit of FastViT or MobileCLIP [5,6,9] and train a
projector to match the teacher's visual representations *and* downstream answers
— a classification or CLIP embedding is not a drop-in for this model's spatial
soft-token sequence. Retain the row count and ordering initially, to isolate
encoder replacement from row-count reduction; supervise on documents, fine text,
diagrams and small objects. It is also the only vision idea that gives up
unchanged-checkpoint parity outright.
**Experiment:** a latency and quality Pareto curve for several students on the
actual target CPU, including preprocessing and projection. Mobile-device or
image-classification results in the cited papers do not prove multimodal CPU
latency here.

R2. **Idea 7 — QAT-cell-certified partial evaluation.** *Hypothesis.
Model-preserving only with sound bounds. No training initially.* Evaluate a
coarse or partial dot product, bound the remaining contribution, and stop when
the **entire** possible result lies inside one output quantization cell; the
quantized value is then fixed without evaluating the remainder. 0.8.9 made the
premise concrete: activations reaching a code plane are exact int8 levels on a
stated step and the kernel computes an exact integer sum, so a bound is a bound
on integers and need not cover the float reduction's own rounding. That removes
the hardest part of the soundness argument and raises the bar — the thing to beat
is now a kernel running at the memory's rate, so a skip must avoid *reads*, not
just multiplies. Candidate bounds: blockwise residual norms or precomputed
absolute weight sums per row, with cheap activation summaries; start with a
projection that immediately applies `leave_gain`. The bound must still cover
clipping, the group scales, ties-to-even and the final requantization.
**Main risk:** high-dimensional bounds are loose, reading enough of a row to
certify the result costs the row, and any precomputed metadata is itself bytes a
bandwidth-bound step must read.
**Experiment and stop rule:** on real activations, histogram how often a cell is
certified and measure **net bytes not read**. Stop if useful skips need most of a
row, or if the metadata cancels the saving. Not in mainline llama.cpp, which has
no early-termination path of any kind.

R3. **Idea 8 — a per-layer-input-conditioned, cache-sharing drafter.**
*Hypothesis. Exact target verification. Drafter training required.* Train a tiny
recurrent or shallow drafter on selected target hidden features and this
architecture's per-layer token inputs, with the objective being agreement with
the **quantized deployed target** rather than low perplexity against a
full-precision teacher. This is what to try when the shipped n-gram scout's
acceptance is poor; keep the verifier, so proposal quality affects speed rather
than correctness. LayerSkip [7] motivates early-exit self-speculation, but
skipping arbitrary layers here is not automatically a trained early-exit model,
and the per-layer embeddings and shared KV owners make the state interface more
involved than a generic shallow transformer. Use only features already available
at the drafting point; running missing target layers to obtain them defeats the
saving.
**Experiment:** compare at equal draft-time budgets against n-gram lookup and an
ordinary small draft model. Track acceptance specifically on image-grounded
answers — a text-only drafter may miss visual facts.

R4. **Idea 3 — T-MAC-style low-bit lookup execution.** *Established.
Model-preserving in intent or approximate depending on the LUT arithmetic. No
training.* Avoid low-bit dequantize-and-multiply by building activation lookup
tables and using packed weight bits to index them [2]. It answers the same
question the integer path answered, so **on a host with an integer dot product
its ceiling is zero** — that host is already at the memory. Its remaining case is
a host with neither AVX-VNNI nor ARM `sdot`, where it is the only route to the
same place, which makes it a portability experiment rather than a headline and
puts it behind item 7 above. Not in mainline llama.cpp; T-MAC is a separate fork.
**Failure modes:** table construction dominates small matrices; lookup precision
changes results; larger tables evict useful data; a bitplane layout needs
packing. It saves instructions, not the weight-byte bandwidth floor.
**Experiment:** total table-build-plus-matmul time and full decode speed on a
host with no integer dot product. Reject isolated kernel wins that lose in the
engine.

R5. **Idea 9 — exploit zeros and changes structurally, not neuron by neuron.**
*Adaptation / hypothesis. Exact zeros or approximate deltas. Training optional.*
Calibrated 8-bit activations can contain exact zeros and skipping their
contributions is mathematically valid, but unstructured indexing may cost more
than dense SIMD, and a transposed or tiled sparse-column strategy must actually
avoid touching the associated weights — the only thing that matters on a
bandwidth-bound step. **The starting position is weaker than it looks:** Gemma 3n
ships an `activation_sparsity_pattern` and llama.cpp implements it
(`src/models/gemma3n.cpp`); **this export has no such field**, so there is no
architectural sparsity here and the idea must earn its keep on measured occupancy
of incidental zeros. PowerInfer [10] is relevant prior work on activation
locality, but its CPU/GPU heterogeneous execution is not CPU-only evidence, and
GELU is not ReLU — near-zero is not zero. The delta variant — cache a previous
projection and compute an activation delta for a similar input — changes the
summation, must use the post-quantization input, and fights the tower's global
attention, where a small pixel change reaches every patch.
**Experiment and stop rule:** histogram exact-zero block occupancy on real
activations **first**. Stop if dense kernels win after indexing, packing and the
extra cached-state traffic — which, at the memory's rate, they very well may.

### Anchors

These support mechanisms, not performance guarantees for igllm. Publication dates
and experimental hardware matter; the newest paper is not necessarily the best
fit. This is a bounded literature pass, not an exhaustive novelty review, and
ideas 5, 7, 8 and the certified/delta combination in 9 are proposed
engine-specific experiments that the sources do not establish as successful here.

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
