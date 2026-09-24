# Provenance: patches/tp3-pad

Written for this repository (2026-09-24) unless noted.

| File | Mounted at | What it is |
|---|---|---|
| tp_pad.py | vllm/models/qwen4_exp/nvidia/tp_pad.py | Original. Load-time padding/replication, plus the vocab-parallel pad. Active only with `QWEN4EXP_TP_PAD=1` |
| test_tp_pad.py | not mounted | Original. Exactness tests against real checkpoint tensors |
| model.py | vllm/models/qwen4_exp/nvidia/model.py | vLLM @ 8a728663 `model.py` + two hooks (`model.diff`) |
| mtp.py | vllm/models/qwen4_exp/nvidia/mtp.py | `../ple-offload/mtp_draft_vocab.py` (upstream reduced-vocab draft, itself vLLM's `mtp.py` + changes) + the TP3 hooks and replicated `fc_embedding`/`fc_hidden` (`mtp.diff`) |

vLLM files are Copyright contributors to the vLLM project, Apache-2.0, and keep their SPDX headers.
