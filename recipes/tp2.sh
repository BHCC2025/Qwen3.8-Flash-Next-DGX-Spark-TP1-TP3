#!/usr/bin/env bash
# recipes/tp2.sh — Qwen3.8-Flash-Next NVFP4 across TWO DGX Sparks (TP2) over one QSFP cable, vLLM mp backend.
# Normally started via ./run.sh tp2 (worker rank 1 first, then the head, rank 0, which serves the API).
#   recipes/tp2.sh 0|1    run one rank on the current node
#
# Settings are the upstream TP2 "speed" profile (tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark, see NOTICE):
# PLE table resident (half per rank), decode CUDA graphs without torch.compile, MTP3, 6 seqs, 4096 chunk, FP8 KV,
# gmu 0.70, 262K. The network is NCCL over RoCE on the one cabled CX7 port of each node (cluster.env TP2_*).
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh"
RANK="${1:?rank 0 or 1}"

PLE_MODE="${PLE_MODE:-none}"                   # CONTEXT profile instead: PLE_MODE=mmap GRAPHS=piecewise MTP=4 SEQS=8 GMU=0.80
GMU="${GMU:-0.70}"; MAXLEN="${MAXLEN:-262144}"; SEQS="${SEQS:-6}"; MTP="${MTP:-3}"
CHUNK="${CHUNK-4096}"
KV_DTYPE="${KV_DTYPE:-fp8_e4m3}"; GRAPHS="${GRAPHS:-nocompile}"; OVERLAYS="${OVERLAYS:-1}"
CAPTURE_SIZES="${CAPTURE_SIZES-}"
MPORT="${MPORT:-29531}"
build_all

DRAFT_ENV=(); DRAFT_MOUNT=()
if [ -n "${DRAFT_VOCAB:-}" ] && [ "$DRAFT_VOCAB" != 0 ]; then
  DRAFT_ENV=(-e QWEN4EXP_DRAFT_VOCAB="$DRAFT_VOCAB")
  DRAFT_MOUNT=(-v "$PLE_DIR/mtp_draft_vocab.py:$NV/mtp.py:ro")
fi

case "$RANK" in
  0) HOST_IP="$TP2_HEAD_IP";   HEADLESS=() ;;
  1) HOST_IP="$TP2_WORKER_IP"; HEADLESS=(--headless) ;;
  *) echo "rank must be 0 or 1" >&2; exit 2 ;;
esac
nccl_env_pair "$RANK"                         # kit/lib/nccl.sh: NCCL over RoCE on the one cabled port

check_model "$MODEL_DIR"
run_container run --gpus all -d --name "$NAME" --restart no \
  --network host --ipc host --shm-size 32g --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  --device /dev/infiniband:/dev/infiniband \
  -v "$MODEL_DIR:/models/qwen38fn:ro" -v "$CACHE_DIR:/root/.cache" \
  -e VLLM_HOST_IP="$HOST_IP" "${BASE_ENV[@]}" "${NCCL_ENV[@]}" \
  "${PLE_ENV[@]}" "${PLE_MOUNT[@]}" "${DRAFT_ENV[@]}" "${DRAFT_MOUNT[@]}" "${OVERLAY_MOUNT[@]}" "${GRAPH_MOUNT[@]}" ${DOCKER_EXTRA:-} \
  "$IMAGE" \
    /models/qwen38fn "${NAME_ARGS[@]}" "${SERVE_ARGS[@]}" --tensor-parallel-size 2 \
    "${SPEC[@]}" "${ASYNC_ARGS[@]}" "${GRAPH_ARGS[@]}" "${KV_ARGS[@]}" \
    --distributed-executor-backend mp --nnodes 2 --node-rank "$RANK" \
    --master-addr "$TP2_HEAD_IP" --master-port "$MPORT" "${HEADLESS[@]}" ${EXTRA:-}
echo "launched $NAME rank=$RANK host=$HOST_IP tp=2 ple=$PLE_MODE graphs=$GRAPHS kv=$KV_DTYPE mtp=$MTP gmu=$GMU maxlen=$MAXLEN"
