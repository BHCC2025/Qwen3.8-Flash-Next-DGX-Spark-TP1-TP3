#!/usr/bin/env bash
# recipes/tp1.sh — Qwen3.8-Flash-Next NVFP4 on ONE DGX Spark (TP1), vLLM. Normally started via ./run.sh tp1.
#
# The whole model (~76 GiB weights + 47.7 GiB PLE n-gram table) does not fit one Spark with room for KV, so the
# table stays on NVMe and its rows are read on demand (PLE_MODE=staged: gathered before each forward, which keeps
# decode CUDA graphs). MTP3 with a 65,536-id reduced draft vocab, 6 seqs, 4096-token prefill chunks, FP8 KV, 262K.
# This is the upstream single-Spark recipe (tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark, see NOTICE) with its
# settings unchanged; only paths and names come from cluster.env.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh"

PLE_MODE="${PLE_MODE:-staged}"
GMU="${GMU:-0.80}"; MAXLEN="${MAXLEN:-262144}"; SEQS="${SEQS:-6}"; MTP="${MTP:-3}"
CHUNK="${CHUNK-4096}"                          # CHUNK= (set, empty) for vLLM's default
KV_DTYPE="${KV_DTYPE:-fp8_e4m3}"; GRAPHS="${GRAPHS:-nocompile}"; OVERLAYS="${OVERLAYS:-1}"
CAPTURE_SIZES="${CAPTURE_SIZES-4,8,12,16,20,24}"  # decode graph widths = (1+MTP) x seqs; needed for SEQS > 4
DRAFT_VOCAB="${DRAFT_VOCAB:-65536}"             # 0 = full-vocab MTP draft
build_all

# Reduced-vocabulary MTP drafting (FR-Spec idea): the draft head scores only the most frequent 65,536 ids.
DRAFT_ENV=(); DRAFT_MOUNT=()
if [ -n "$DRAFT_VOCAB" ] && [ "$DRAFT_VOCAB" != 0 ]; then
  DRAFT_ENV=(-e QWEN4EXP_DRAFT_VOCAB="$DRAFT_VOCAB")
  DRAFT_MOUNT=(-v "$PLE_DIR/mtp_draft_vocab.py:$NV/mtp.py:ro")
fi

check_model "$MODEL_DIR"
run_container run --gpus all -d --name "$NAME" --restart no \
  --network host --ipc host --shm-size 32g --ulimit memlock=-1:-1 \
  -v "$MODEL_DIR:/models/qwen38fn:ro" -v "$CACHE_DIR:/root/.cache" \
  "${BASE_ENV[@]}" "${PLE_ENV[@]}" "${PLE_MOUNT[@]}" "${DRAFT_ENV[@]}" "${DRAFT_MOUNT[@]}" \
  "${GRAPH_MOUNT[@]}" "${OVERLAY_MOUNT[@]}" ${DOCKER_EXTRA:-} \
  "$IMAGE" \
    /models/qwen38fn "${NAME_ARGS[@]}" "${SERVE_ARGS[@]}" --tensor-parallel-size 1 \
    "${SPEC[@]}" "${ASYNC_ARGS[@]}" "${GRAPH_ARGS[@]}" "${KV_ARGS[@]}" ${EXTRA:-}
echo "launched $NAME tp=1 ple=$PLE_MODE graphs=$GRAPHS kv=$KV_DTYPE mtp=$MTP draft_vocab=$DRAFT_VOCAB gmu=$GMU maxlen=$MAXLEN seqs=$SEQS"
