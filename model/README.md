# model

`google/gemma-4-E2B-it-qat-mobile-transformers`, vendored so the engine can be
built and checked on a machine that cannot reach Hugging Face.

    python3 run.py run -- chat --model model --prompt "Explain gravity to a child."
    python3 run.py check  --model model
    python3 run.py parity --model model
    python3 run.py parity --media --model model --image photo.png --audio clip.wav

## What is here

The upstream repository ships one `model.safetensors` of 2.4 GB. That is over
GitHub's two gigabyte limit for a single LFS object, so it is split into three
shards with the index that names them — the ordinary Hugging Face layout, which
both this engine and `transformers` read without being told. The split is byte
exact: only the offsets in each header were rebased, so no value moved and no
dtype was round-tripped.

| file | what it holds |
| ---- | ------------- |
| `model-0000N-of-00003.safetensors` | the weights, 2780 tensors over three shards |
| `model.safetensors.index.json` | which shard holds which tensor |
| `config.json` | the text stack, and the two tower blocks |
| `preprocessor_config.json` | the audio analysis window the filterbank reads |
| `processor_config.json` | the image and audio token budgets |
| `tokenizer.json` | the vocabulary and merges |
| `generation_config.json`, `tokenizer_config.json`, `chat_template.jinja` | as upstream |

The shards are Git LFS objects; see `.gitattributes` at the repository root. A
clone without LFS installed will find text pointers here rather than weights, so
run `git lfs install` before cloning, or `git lfs pull` after.

## Licence

The weights are Google's, released under the Apache License 2.0 with the
[Gemma 4 terms](https://ai.google.dev/gemma/docs/gemma_4_license). They are
redistributed here unmodified apart from the sharding described above. The
engine in this repository is a separate work and carries its own terms.
