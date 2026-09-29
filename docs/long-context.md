# 1M context (TP3)

The model is trained to 262,144 tokens. Qwen's model card extends it to ~1M with **static YaRN, factor 4**, applied
to the text `rope_parameters` (the multimodal mrope sections are kept).

```bash
scripts/prep-tp3-modeldir.sh --longctx     # builds MODEL_DIR_TP3_1M
./run.sh tp3-1m                            # MAXLEN=1000000
```

Static means the scaling applies to every prompt, short ones included. Qwen notes this can cost a little quality on
short prompts, so it's a separate mode rather than the default. In our short-prompt runs (2026-09-29) it decoded
57.6 / 39.6 tok/s code / prose, against 62.8 / 40.7 without YaRN.

## Why it's in config.json, not `--hf-overrides`

vLLM applies dict `--hf-overrides` to the target model only. The MTP draft model keeps 262,144, fails vLLM's
`mamba_block_size` consistency check, and its RoPE would stop at 262,144 anyway. Writing YaRN into `config.json`
means the target and the draft both see it.

The top-level config also gets its own default `rope_parameters`. Without that, transformers copies the text YaRN to
the top level and validates it there, where there's no `max_position_embeddings`, and loading fails.

## Measured (3 Sparks, 2026-09-29)

Needle test (`kit/bench/bench-longctx.py`, run by `LONG=1 bench/bench.sh`): three passphrases at 25/50/75 % depth,
cold prefill each time. Full results: [bench/results/2026-09-29-tp3.md](../bench/results/2026-09-29-tp3.md).

| Prompt tokens | Time to first token | Prefill | Needles |
|---|---|---|---|
| 128,278 | 55 s | 2,324 tok/s | 3/3 |
| 256,580 | 116 s | 2,205 tok/s | 3/3 |
| 512,460 | 247 s | 2,071 tok/s | 3/3 |
| 899,097 | 467 s | 1,923 tok/s | 3/3 |
| 988,114 | 525 s | 1,882 tok/s | 3/3 |

KV pool at `GMU=0.70` on the 1M server: 2,462,621 tokens (vLLM's startup log), so a full 1M request fits alongside
other traffic.
