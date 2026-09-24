# TP3: padding the model so it splits three ways

Tensor parallelism splits heads and MLP widths evenly across ranks. For Qwen3.8-Flash-Next, three of those
dimensions don't divide by 3:

| Dimension | Checkpoint | Padded | How | Why it's exact |
|---|---|---|---|---|
| Full-attention KV heads | 2 | 6 | each KV head copied 3× | query head *h* reads KV head *h*//4, a copy of original KV head *h*//12 (which is what it read before) |
| GatedDeltaNet q/k heads | 16 | 48 | each q/k head (and its conv channels) copied 3× | value head *i* reads key head *i*, a copy of original key head *i*//3 |
| Routed + shared expert intermediate | 640 | 768 | zero rows in gate/up, zero columns in down | silu(0)·0 = 0 and zero down-columns add nothing; 768/3 = 256 per rank (2 FP8 blocks, 16 NVFP4 groups) |
| Vocab-parallel embeddings / LM head | 248,320 | 248,448 | pad to a multiple of 64 × TP | the extra rows are zero and vLLM trims logits back to the real vocab |

Everything else (the indexer, hyper-connections, the n-gram table, embeddings) is either already replicated or
already divisible. The MTP head's `fc_embedding`/`fc_hidden` (output width 2,560) is made replicated.

## Pieces

- `scripts/prep-tp3-modeldir.sh` builds `MODEL_DIR_TP3`: hardlinks to the checkpoint plus a `config.json` with the
  four padded sizes. vLLM reads the config and allocates the padded parameter shapes.
- `patches/tp3-pad/tp_pad.py` transforms each checkpoint tensor as it streams into `load_weights`, and patches
  `VocabParallelEmbedding` to pad to 64 × TP. It does nothing unless `QWEN4EXP_TP_PAD=1`, which `recipes/tp3.sh` sets.
- `patches/tp3-pad/model.py`, `mtp.py` are the vLLM nightly's files with the hooks added. The `.diff` files beside
  them show exactly what changed.
- `patches/tp3-pad/test_tp_pad.py` checks against real checkpoint tensors: identical GQA output, identical
  GatedDeltaNet head mapping per rank, identical shared-expert MLP output, and zero padding everywhere else.

The FP8 block scales of the padded MTP expert blocks are set to 1.0, not 0. An all-zero block with a zero scale
tripped a re-encode path in a similar TP3 port; 0 × 1 = 0 either way.

## Running the tests

```bash
docker run --rm --gpus all -v "$MODEL_DIR:/models/qwen38fn:ro" -v "$PWD/patches/tp3-pad:/work" \
  --entrypoint python3 vllm/vllm-openai:nightly-8a728663c1c3eeace834a95f5654fa653cc1998c /work/test_tp_pad.py
```

Expected: every line `PASS`, ending in `ALL PASS`.

## Cost

Replicating KV heads triples the full-attention layers' KV cache per token (GatedDeltaNet layers keep a fixed-size
state instead). That still leaves a large
pool: about 2.47M tokens at `GMU=0.70` on three Sparks. Everything else is unchanged or zero padding.
