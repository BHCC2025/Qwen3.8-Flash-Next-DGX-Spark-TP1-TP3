# Provenance: patches/tp3-pad

Written for this repository (2026-09-24) unless noted.

| File | Mounted at | What it is | sha256 |
|---|---|---|---|
| tp_pad.py | vllm/models/qwen4_exp/nvidia/tp_pad.py | Original. Load-time padding/replication, plus the vocab-parallel pad. Active only with `QWEN4EXP_TP_PAD=1` | 18c74438724fa343a4d0ee1f92880a76c1a33aa830eff92d77e30387757e88d0 |
| test_tp_pad.py | not mounted | Original. Exactness tests against real checkpoint tensors | 52093138864265799bca77d81ef3212811683e7c293c99806786869822f913b5 |
| model.py | vllm/models/qwen4_exp/nvidia/model.py | vLLM @ 8a728663 `model.py` + two hooks (`model.diff`) | d0081a2fc124e9fd58276b1a9be0366a8c8a3284795881d874f942e8c3b9dae8 |
| mtp.py | vllm/models/qwen4_exp/nvidia/mtp.py | `../ple-offload/mtp_draft_vocab.py` (upstream reduced-vocab draft, itself vLLM's `mtp.py` + changes) + the TP3 hooks and replicated `fc_embedding`/`fc_hidden` (`mtp.diff`) | 38ecfc8b09d861607d157c51f28c3b1f1efc3585e9cd48b85e3118e4e89dacf1 |

vLLM files are Copyright contributors to the vLLM project, Apache-2.0, and keep their SPDX headers.
