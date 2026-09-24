# Provenance: patches/ple-offload

Vendored from [tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark](https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark)
at commit 6ad1c8f15cbab1ababd2048e8e5f94094dbfc4a0, directory `single-spark-vllm-tp1/patch/` (Apache-2.0).
Every file is byte-identical to that commit except one:

| File | Change here | sha256 (this repo) |
|---|---|---|
| upstream-overlays/modelopt.py | `FP8_PB_WO` added to the block-FP8 MoE algo list (one line, `modelopt_fp8_pb_wo.diff`). Checkpoint revision fc694b54 names the MTP experts' algo `FP8_PB_WO`; the same alias was added upstream in vLLM PR #55513. | 0d8b239befc8b2348c6249b08b794800c40ca6003e9eda09560a2177e0597959 |

Upstream's README and `upstream-overlays/PROVENANCE.md` (kept here unchanged) document what each file does and who
wrote which change. In short: `ple_mmap.py`, `ple_layer.py`, `model_state.py` and `compilation.py` keep the
47.7 GiB n-gram table on NVMe and gather rows on demand; `mtp_draft_vocab.py` is the reduced-vocabulary MTP draft;
`upstream-overlays/` carries vLLM PRs #55375 and #54846 plus the two MTP-loading fixes to `modelopt.py`.
