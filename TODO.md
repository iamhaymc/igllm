# TODO

Open development tasks. The first group can only be judged where the shipped
export is — against the real weights, or against the export's own configuration
read by the reference's own code — so it belongs on a host that has `model/`
beside it. The second group can be written and judged anywhere, on synthetic
weights or on none. Most consequential first within each group.

## On the shipped export

- Spend fewer instructions in the decode kernels. This is where the decode time
  goes, measured rather than argued in 0.8.1: on four cores the tuned build
  reads at 10.55 gigabytes a second against the 24.87 the host's memory will
  hand over on a bare sweep, and from one thread to four it produces 3.16 times
  as many tokens where the memory gives 1.54 times as much. Neither number is
  the shape of a run against the memory. The candidate below it — a per-plane
  float mirror of the group gains, trading memory for a shorter inner loop — is
  open on both builds again; 0.8.0 struck it from the tuned build on the reading
  0.8.1 withdrew. What has to be weighed is the bytes it adds against the
  conversions it saves, and now on the tuned build as well as the default.
- Read fewer bytes a token. What a decode step reads is 759.4 MiB against the
  2334.8 MiB the export maps, and 0.8.1's table says where it goes: 475.3 in the
  mlp projections, 97.0 in the output head, 126.5 in attention. The two thirds
  of the mapped total a token never touches — the per-layer embedding table at
  1120 MiB, the two towers at 324 — are a footprint question rather than a decode
  one, and letting the caller choose residency per tensor is still the knob for
  that. It is worth having on a small host; it is not the decode win 0.8.0 took
  it for.
- Hold the seam open as the prompt grows. All nine cases are judged on the
  shipped export now, each in a process of its own, and the longest is 664 ids
  with two pictures and a clip in it. What has not been tried is a prompt long
  enough that the reference's forward stops fitting even alone — the cost is the
  reference's rather than the engine's, and the answer when it comes is probably
  to compare against a cached forward rather than a live one.
- Speed up the towers further. Vision attention runs a head at a time over the
  pool, with each head's keys and values gathered into a run that stays in
  cache, and the eight bit projections the tower is quantized to now have a
  vector path: a whole run — an image at the full patch budget, and the 316 ids
  of prefill behind it — went from four minutes twenty-eight to one minute two
  on the SSE2 build. What is left: the score and blend loops are scalar where
  the dot product has a vector path, an image at the full patch budget still
  costs six hundred billion multiply-adds however it is arranged, and the
  conformer's convolution module still walks its kernel per channel per frame.
  None of it runs in the token loop, and a prompt may now carry several of
  them, which makes it the cost of the first answer rather than of the run.
- Find out whether the gather is the wrong read for the byte cache. 0.8.3 counts
  what the byte cache saves — 48.3 MiB a token as floats against 12.1 as bytes
  on a 694 id prompt, a quarter — but could not time it: the host it was written
  on decoded the same build in the same configuration at 4.52 tok/s in one
  window and 2.32 in another. What the five pairs there do agree on is a sign,
  and only on the tuned build: all three tuned pairs put the byte cache a third
  to a half behind, where the two default pairs disagree about the direction.
  Three of one sign is not a measurement, but it is what the trade predicts —
  `cache_dot` spends a `vgatherdps` per eight values to save three quarters of
  the cache traffic, on a build 0.8.1 showed is not against the memory. If a
  quiet host confirms it, the answer is a different read rather than a faster
  one: the codes are four to a dword and a shuffle-based spread may beat the
  gather outright, which would serve the SSE2 and NEON paths too. This is the
  gain mirror's trade above taken in the other direction, and the answer for one
  informs the other.
- Cache dequantized scales for the hottest planes. Gains are converted from
  their stored dtype on every group; a per-plane float mirror trades memory for
  a shorter inner loop. 0.8.0 struck this from the tuned build on the reading
  0.8.1 withdrew, and it is open on both builds again: the tuned build is at 42%
  of what the memory gives and the default at 30%, so on neither of them is the
  read the whole cost. What has to be weighed is the bytes the mirror adds
  against the conversions it saves, and the two builds will not answer the same
  way.

## Anywhere

- Read a wider range of pictures. The png reader refuses interlaced files and
  bit depths under eight, and there is no jpeg reader at all.
- Support more than one concurrent session per model in the CLI, and add a
  multi-turn chat loop rather than a single turn.
- Persist and restore a session cache, so a long prompt need not be primed
  twice.
- Widen `kern_dot_code` for the odd bit widths. Two, four and eight bits are
  vectorized, in the fused dot and in `kern_code_spread` beside it; three,
  five, six, and seven still walk the bit stream in both. Synthetic weights
  reach every width, which the shipped export does not.
- Add an AVX-512 path beside AVX2, selected by the same macro layer. It cannot
  be judged on a host without the instruction set, and the 2017 desktop the
  numbers above were taken on has none. On the evidence above it would have
  something to show rather than nothing: the tuned build uses 42% of what the
  memory gives, so a wider kernel has room to move the token rate.
- Add a device handle beside `plane` and a second `back_open`, so an
  accelerator backend can be dropped in without touching the loader.
- Build under MSVC. The suite builds clean and passes on a Windows host with
  MinGW gcc, on the scalar, SSE2 and AVX2 backends; `cl` and its `/arch:AVX2`
  path have still only been read.
